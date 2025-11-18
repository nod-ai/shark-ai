# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import logging
import json
import time
from tqdm import tqdm
from typing import Any

import torch

from amdsharktank.models.llm import *

from amdsharktank.layers import *
from amdsharktank.types import *
import amdsharktank.ops as ops

from amdsharktank.utils import cli
from amdsharktank.utils.load_llm import *
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.create_cache import *
from amdsharktank.utils.export_artifacts import *
from amdsharktank.utils.evaluate import *
from amdsharktank.utils.iree import *

import iree.runtime as ireert

logger = get_logger(__name__)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)

__all__ = ["PerplexityIree", "run_perplexity_iree"]


class PerplexityIree:
    """
    Perplexity (PPL) is one of the most common metrics for evaluating language models.
    It is defined as the exponentiated average negative log-likelihood of a sequence,
    calculated with exponent base `e`.

    For more information, see https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        torch_device,
        iree_devices: list[str],
        iree_hip_target,
        iree_hal_target_device,
        bs,
        tensor_parallelism_size,
        pipeline_parallelims_size,
        attention_kernel,
        matmul_kernel,
        block_seq_stride,
        activation_dtype,
        attention_dtype,
        kv_cache_dtype,
        interleave_rotary: bool,
        weight_path_str: str,
        prefill_length: int | None = None,
        use_toy_model: bool = False,
        extra_compile_args: list[str] | None = None,
    ):
        self.torch_device = torch_device
        self.iree_devices = iree_devices
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_device = iree_hal_target_device
        self.bs = bs
        self.tensor_parallelism_size = tensor_parallelism_size
        self.attention_kernel = attention_kernel
        self.matmul_kernel = matmul_kernel
        self.block_seq_stride = block_seq_stride
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.pipeline_parallelism_size = pipeline_parallelims_size
        self.interleave_rotary = interleave_rotary
        self.weight_path_str = weight_path_str
        assert prefill_length is None or prefill_length >= 1
        self.prefill_length = prefill_length
        self.use_toy_model = use_toy_model
        self.extra_compile_args = extra_compile_args
        self.vm_context: iree.runtime.VmContext = None
        self.cache_state: None | list[ireert.DeviceArray] = None

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time_ns()
            result = func(*args, **kwargs)
            end = time.time_ns()
            time_taken = calc_time(start, end)
            func_name = func.__name__
            logger.info(f" {func_name}: {time_taken}")
            return result

        return wrapper

    def print_token_comparison(self, i: int):
        if self.use_toy_model and i <= self.max_prompt_length:
            batch_predicted_token_id = [[i[-1]] for i in self.batch.results]
            logger.debug(f"Predicted:")
            logger.debug(f"{batch_predicted_token_id}")

            expected_token_id = self.token_ids[:, i + 1 : i + 2].tolist()
            logger.debug(f"Expected:")
            logger.debug(f"{expected_token_id}")

        elif i <= self.max_prompt_length:
            batch_predicted_token_id = [[i[-1]] for i in self.batch.results]
            batch_predicted_token = self.generator.tokenizer.decode(
                batch_predicted_token_id
            )
            logger.debug(f"Predicted:")
            logger.debug(f"{batch_predicted_token}")
            logger.debug(f"{batch_predicted_token_id}")

            expected_token_id = self.token_ids[:, i + 1 : i + 2].tolist()
            expected_token = self.generator.tokenizer.decode(expected_token_id)
            logger.debug(f"Expected:")
            logger.debug(f"{expected_token}")
            logger.debug(f"{expected_token_id}")

    def export_compile_model(
        self,
        output_mlir: str | None,
        output_config: str | None,
        output_vmfb: str | None,
    ):
        logger.info(f" Model: {self.weight_path_str}")

        cwd = (
            Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent
            / "perplexity_ci_artifacts/"
        )

        activation_dtype = (
            str(self.activation_dtype).split(".")[-1] if self.activation_dtype else None
        )
        attention_dtype = (
            str(self.attention_dtype).split(".")[-1] if self.attention_dtype else None
        )
        kv_cache_dtype = (
            str(self.kv_cache_dtype).split(".")[-1] if self.kv_cache_dtype else None
        )

        export_artifacts = ExportArtifacts(
            irpa_path=self.weight_path_str,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            hip_device_id=self.iree_devices[0],
            attention_kernel=self.attention_kernel,
            matmul_kernel=self.matmul_kernel,
            tensor_parallelism_size=self.tensor_parallelism_size,
            pipeline_parallelism_size=self.pipeline_parallelism_size,
            block_seq_stride=self.block_seq_stride,
            activation_dtype=activation_dtype,
            attention_dtype=attention_dtype,
            kv_cache_dtype=kv_cache_dtype,
            interleave_rotary=self.interleave_rotary,
            output_mlir=output_mlir,
            output_config=output_config,
            output_vmfb=output_vmfb,
            cwd=cwd,
        )
        self.output_vmfb = export_artifacts.export_and_compile_llm(
            batch_size=self.bs, extra_compile_args=self.extra_compile_args
        )

    @timeit
    def load_model(
        self, dataset: Dataset, tokenizer: Optional[InferenceTokenizer] = None
    ):
        assert self.pipeline_parallelism_size == 1
        assert self.tensor_parallelism_size == 1

        config = LlamaModelConfig.from_dataset(
            dataset=dataset,
            device=self.torch_device,
            activation_dtype=self.activation_dtype,
            attention_dtype=self.attention_dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            block_seq_stride=self.block_seq_stride,
            attention_kernel=self.attention_kernel,
            matmul_kernel=self.matmul_kernel,
        )

        hp = config.hp

        config.parallelism_config = ParallelismConfig.default_config(
            block_count=hp.block_count,
            pp=self.pipeline_parallelism_size,
            tp=self.tensor_parallelism_size,
        )

        theta = dataset.root_theta

        model = PagedLlmModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

        shard_count = self.tensor_parallelism_size

        self.devices: list[iree.runtime.HalDevice] = get_iree_devices(
            device=self.iree_devices,
            device_count=self.pipeline_parallelism_size * shard_count,
            allow_repeating=True,
        )

        self.vm_module, self.vm_context, self.vm_instance = load_iree_module(
            module_path=self.output_vmfb,
            devices=self.devices,
            parameters_path=self.weight_path_str,
            tensor_parallel_size=shard_count,
            pipeline_parallel_size=self.pipeline_parallelism_size,
        )

    def assemble_batch(self, token_batch: torch.tensor, devices) -> torch.tensor:

        token_batch, seq_lens_batch = pad_tokens(
            token_ids=token_batch.tolist(),
            pad_to_multiple_of=self.generator.model.config.block_seq_stride,
        )

        logger.debug(f"{token_batch}")

        token_batch = torch.as_tensor(token_batch, device=self.torch_device)
        seq_lens_batch = torch.as_tensor(seq_lens_batch, device=self.torch_device)

        self.batch = self.generator.begin_batch(
            token_ids=token_batch,
            seq_lens=seq_lens_batch,
            page_cache_size=self.page_cache_size,
            max_decode_steps=self.max_prompt_length - self.prefill_length - 1,
        )

        self.cache_state = []
        for i in range(self.pipeline_parallelism_size):
            self.cache_state.extend(
                prepare_iree_module_function_args(
                    args=[self.batch.cache_state[i]],
                    devices=devices[i : (i + 1) * self.tensor_parallelism_size],
                )
            )

        return token_batch

    def prefill_vmfb(
        self, token_batch: torch.tensor, i: int, devices: list[iree.runtime.HalDevice]
    ) -> torch.tensor:
        if not self.use_toy_model:
            logger.debug(
                f"Prefill input:\n\t\t   {self.generator.tokenizer.decode(token_batch)}"
            )

        token_batch = self.assemble_batch(token_batch, devices)

        prefill_kwargs = OrderedDict(
            [
                ("tokens", token_batch),
                ("seq_lens", [self.batch.seq_lens]),
                ("seq_block_ids", [self.batch.pad_block_ids()]),
                ("cache_state", self.cache_state),
            ]
        )
        prefill_kwargs_flattened = flatten_for_iree_signature(prefill_kwargs)
        # NOTE: cache_state must be handled previous to this since
        #       prepare_iree_module_function_args is unable to know about
        #       pipeline parallelism.
        prefill_iree_args = prepare_iree_module_function_args(
            args=prefill_kwargs_flattened, devices=devices
        )

        prefill_iree_result = run_iree_module_function(
            args=prefill_iree_args,
            function_name=f"prefill_bs{self.bs}",
            module=self.vm_module,
            vm_context=self.vm_context,
            device=devices[0],
        )
        prefill_shards = iree_to_torch(*prefill_iree_result)
        if self.tensor_parallelism_size > 1:
            prefill_logits = ops.unshard(UnreducedTensor(ts=prefill_shards))
        else:  # Replicated or torch.Tensor
            prefill_logits = prefill_shards[0]
        prefill_logits = prefill_logits.clone().detach()

        tokens = torch.as_tensor(
            self.generator.model.extract_tokens_from_logits(
                prefill_logits, self.batch.seq_lens
            )
        ).unsqueeze(1)
        self.batch.add_result_token(tokens)

        self.print_token_comparison(i)
        return prefill_logits

    def decode_vmfb(
        self, token_batch: torch.tensor, i: int, devices: list[iree.runtime.HalDevice]
    ) -> torch.tensor:
        logger.debug(f"Decode input:")
        if not self.use_toy_model:
            logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")
        logger.debug(f"{token_batch.tolist()}")

        start_positions = [self.batch.seq_lens.clone()]
        self.batch.seq_lens.add_(1)
        self.batch.allocate_seq_block_ids()

        decode_kwargs = OrderedDict(
            [
                ("tokens", token_batch),
                ("seq_lens", [self.batch.seq_lens]),
                (
                    "start_positions",
                    start_positions,
                ),
                ("seq_block_ids", [self.batch.pad_block_ids()]),
                ("cache_state", self.cache_state),
            ]
        )
        decode_kwargs_flattened = flatten_for_iree_signature(decode_kwargs)
        decode_iree_args = prepare_iree_module_function_args(
            args=decode_kwargs_flattened, devices=devices
        )
        decode_iree_result = run_iree_module_function(
            args=decode_iree_args,
            function_name=f"decode_bs{self.bs}",
            module=self.vm_module,
            vm_context=self.vm_context,
            device=devices[0],
        )

        decode_shards = iree_to_torch(*decode_iree_result)
        if self.tensor_parallelism_size > 1:
            decode_logits = ops.unshard(UnreducedTensor(ts=decode_shards))
        else:  # Replicated or torch.Tensor
            decode_logits = decode_shards[0]
        decode_logits = torch.as_tensor(decode_logits[:, :, :])

        tokens = torch.as_tensor(
            self.generator.model.extract_tokens_from_logits(
                decode_logits, [1] * self.bs
            ),
            device=self.generator.model.device,
        ).unsqueeze(1)
        self.batch.add_result_token(tokens)

        self.print_token_comparison(i)
        return decode_logits

    def get_logits(self, skip_decode: bool) -> torch.Tensor:
        self.prefill_time = 0
        self.decode_time = []

        def run_iree_module(devices: list[iree.runtime.HalDevice]):
            out_logits = []
            model_name = Path(self.weight_path_str).name
            for i in tqdm(
                range(self.prefill_length - 1, self.max_prompt_length - 1),
                mininterval=300,
                desc=f"eval_iree: Calculating logits for {model_name}",
            ):
                logger.debug(f"Iteration: {i - self.prefill_length + 1}")

                if skip_decode or len(out_logits) == 0:
                    token_batch = self.token_ids[:, : i + 1]

                    start = time.time_ns()
                    prefill_logits = self.prefill_vmfb(token_batch, i, devices).clone()
                    self.prefill_time = time.time_ns() - start

                    last_logits_indices = torch.minimum(
                        self.seq_lens - 1, torch.tensor(i)
                    )
                    last_logits_indices = torch.maximum(
                        last_logits_indices, torch.tensor(0)
                    )
                    last_real_prefill_logits = prefill_logits[
                        self.batch_indices, last_logits_indices, :
                    ].unsqueeze(1)
                    out_logits.append(last_real_prefill_logits)
                else:
                    token_batch = self.token_ids[:, i : i + 1]
                    start = time.time_ns()
                    decode_logits = self.decode_vmfb(token_batch, i, devices)
                    self.decode_time.append(time.time_ns() - start)
                    out_logits.append(decode_logits)

            out_logits = ops.cat(out_logits, dim=1)

            pad_logits_shape = self.token_ids.shape[1] - out_logits.shape[1]

            pad_logits = torch.zeros(
                out_logits.shape[0], pad_logits_shape, out_logits.shape[2]
            )

            self.cache_state = None  # Remove saved reference to iree.runtime.DeviceArray before leaving function
            return ops.cat(
                (
                    pad_logits[:, : self.prefill_length],
                    out_logits,
                    pad_logits[:, self.prefill_length :],
                ),
                dim=1,
            ).to(self.torch_device)

        return with_iree_device_context(run_iree_module, self.devices)

    @timeit
    def get_perplexity(
        self, test_prompts: list[str], token_ids: list[list[int]], skip_decode: bool
    ) -> dict[str, Any]:

        if self.use_toy_model:
            self.token_ids = token_ids
            self.seq_lens = [len(t) for t in self.token_ids]
            # Add context to improve perplexity by starting at 5th token
            if self.prefill_length is None:
                self.prefill_length = 6
            self.page_cache_size = 128
            logger.debug(f" Token ids for Evaluation: \n{self.token_ids}\n")

        else:
            self.token_ids, self.seq_lens = self.generator.tokenizer.encode(
                test_prompts,
                pad_to_multiple_of=self.generator.model.config.block_seq_stride,
            )

            logger.debug(f" Prompts for Evaluation:")
            for idx, prompt in enumerate(test_prompts):
                logger.debug(
                    f" Prompt {idx}: \nTokens: {prompt.encode()}\nToken ids: {self.token_ids[idx]}\n"
                )

            # Add context to improve perplexity by starting at 10th token
            if self.prefill_length is None:
                self.prefill_length = 11
            self.page_cache_size = (
                len(self.token_ids[0]) // self.generator.model.config.block_seq_stride
            ) * len(test_prompts) + 1

        self.max_prompt_length = max(self.seq_lens)

        context_length = self.generator.model.config.hp.context_length
        if self.max_prompt_length > context_length:
            logger.warning(
                f"Last token {self.max_prompt_length} exceeds context length {context_length}. "
                "Limiting tokens to context length."
            )
            self.max_prompt_length = context_length

        self.token_ids = torch.as_tensor(self.token_ids, device=self.torch_device)
        self.seq_lens = torch.tensor(self.seq_lens, device=self.torch_device)

        self.batch_indices = torch.arange(len(self.seq_lens))

        out_logits = self.get_logits(skip_decode)

        logger.debug(f"Final Logits shape: {out_logits.shape}")
        logger.debug(f"Token ids shape: {self.token_ids.shape}")

        return compute_perplexity(
            self.token_ids, out_logits, self.prefill_length - 1, self.max_prompt_length
        )


def run_perplexity_iree(
    args,
    dataset: Dataset,
    tokenizer: InferenceTokenizer,
    torch_device: torch.device,
    tensor_parallelism_size: int,
    pipeline_parallelism_size: int,
) -> dict[str, Any]:
    start = time.time_ns()

    token_ids = None
    test_prompts = None

    if args.use_toy_model:
        token_ids = get_token_ids()
        bs = len(token_ids)
    else:
        test_prompts = args.prompt_list or get_prompts(num_prompts=args.num_prompts)
        bs = len(test_prompts)

    perplexity = PerplexityIree(
        torch_device=torch_device,
        iree_devices=args.iree_device,
        iree_hip_target=args.iree_hip_target,
        iree_hal_target_device=args.iree_hal_target_device,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelims_size=pipeline_parallelism_size,
        attention_kernel=args.attention_kernel,
        matmul_kernel=args.matmul_kernel,
        block_seq_stride=args.block_seq_stride,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        interleave_rotary=not args.use_hf,
        bs=bs,
        weight_path_str=str(args.irpa_file),
        prefill_length=args.prefill_length,
        use_toy_model=args.use_toy_model,
        extra_compile_args=args.extra_compile_arg,
    )

    perplexity.export_compile_model(
        output_mlir=args.output_mlir,
        output_config=args.output_config,
        output_vmfb=args.output_vmfb,
    )
    perplexity.load_model(
        dataset=dataset,
        tokenizer=tokenizer,
    )
    perplexity_batch = perplexity.get_perplexity(
        test_prompts=test_prompts,
        token_ids=token_ids,
        skip_decode=args.skip_decode,
    )

    logger.info(f" Total time taken: {calc_time(start, time.time_ns())}")
    logger.info(f" Prefill time: {calc_time(time_diff=perplexity.prefill_time)}")
    if not args.skip_decode:
        decode_time = sum(perplexity.decode_time) / len(perplexity.decode_time)
        logger.info(f" Decode time per token: {calc_time(time_diff=decode_time)}")

    return {
        "perplexities": perplexity_batch,
        "mean_perplexity": round(np.mean(perplexity_batch), 6),
    }


def main(argv):
    parser = cli.create_parser()

    cli.add_evaluate_options(parser)
    cli.add_export_artifacts(parser)
    cli.add_iree_flags(parser)
    cli.add_model_options(parser)
    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    cli.add_log_options(parser)

    args = cli.parse(parser, args=argv)
    dataset = cli.get_input_dataset(args)
    tokenizer = None
    if not args.use_toy_model:
        tokenizer = cli.get_tokenizer(args)

    logger.setLevel(args.loglevel)
    torch_device = torch.device(args.device) if args.device else None

    assert args.num_prompts or args.prompt_list, "Pass --num-prompts or --prompt-list"

    if args.output_mlir or args.output_config:
        assert (
            args.output_config is not None and args.output_mlir is not None
        ), "If using pre-exported mlir, both --mlir-path and --json-path must be passed"

    # Ensure tensor parallelism flag agrees with dataset properties
    if "tensor_parallelism_size" in dataset.properties:
        dataset_tensor_parallelism_size = dataset.properties["tensor_parallelism_size"]
        if dataset_tensor_parallelism_size != args.tensor_parallelism_size:
            raise ValueError(
                f"Tensor parallelism size mismatch: dataset={dataset_tensor_parallelism_size} while arg={args.tensor_parallelism_size}. Wrong value for --tensor-parallelism-size."
            )
    else:
        if args.tensor_parallelism_size != 1:
            raise ValueError(
                f"Unsharded dataset file provided, but specified --tensor-parallelism-size={args.tensor_parallelism_size}. Likely wrong dataset provided."
            )

    ppl = run_perplexity_iree(
        args,
        dataset=dataset,
        tokenizer=tokenizer,
        torch_device=torch_device,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
