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

import torch
import iree.runtime as ireert

from sharktank.models.llm import *

from sharktank.layers import *
from sharktank.types import *
import sharktank.ops as ops

from sharktank.utils import cli
from sharktank.utils.load_llm import *
from sharktank.utils.create_cache import *
from sharktank.utils.export_artifacts import *
from sharktank.utils.evaluate import *
from sharktank.utils.iree import *

logger = logging.getLogger("eval")

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
        block_seq_stride,
        activation_dtype,
        attention_dtype,
        kv_cache_dtype,
        use_attention_mask,
        use_hf,
        weight_path_str: str,
    ):
        self.torch_device = torch_device
        self.iree_devices = iree_devices
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_device = iree_hal_target_device
        self.bs = bs
        self.tensor_parallelism_size = tensor_parallelism_size
        self.attention_kernel = attention_kernel
        self.block_seq_stride = block_seq_stride
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.pipeline_parallelism_size = pipeline_parallelims_size
        self.attention_kernel = attention_kernel
        self.use_attention_mask = use_attention_mask
        self.use_hf = use_hf
        self.weight_path_str = weight_path_str
        self.vm_context: iree.runtime.VmContext = None
        self.cache_state: None | list[ireert.DeviceArray] = None

    def print_token_comparison(self, i: int):
        if i <= self.max_prompt_length:
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

    def compile_model(
        self,
        output_mlir: str,
        output_config: str,
        output_vmfb: str,
    ):

        logger.info(f" Model: {self.weight_path_str}")

        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = self.attention_dtype

        if output_vmfb:
            self.output_vmfb = output_vmfb
            logger.info(f" Using pre-compiled vmfb: {self.output_vmfb}")
        else:
            export_artifacts = ExportArtifacts(
                irpa_path=self.weight_path_str,
                batch_size=self.bs,
                iree_hip_target=self.iree_hip_target,
                iree_hal_target_device=self.iree_hal_target_device,
                attention_kernel=self.attention_kernel,
                tensor_parallelism_size=self.tensor_parallelism_size,
                pipeline_parallelism_size=self.pipeline_parallelism_size,
                block_seq_stride=self.block_seq_stride,
                use_attention_mask=self.use_attention_mask,
                activation_dtype=str(self.activation_dtype).split(".")[-1],
                attention_dtype=str(self.attention_dtype).split(".")[-1],
                kv_cache_dtype=str(self.kv_cache_dtype).split(".")[-1],
                use_hf=self.use_hf,
                output_mlir=output_mlir,
                output_config=output_config,
            )
            self.output_vmfb = export_artifacts.get_artifacts()

    def load_model(self, dataset: Dataset, tokenizer: InferenceTokenizer):
        hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
        block_to_device_lookup = []
        for i in range(hp.block_count):
            pp_group = int(i * self.pipeline_parallelism_size / hp.block_count)
            zero_4_group = self.tensor_parallelism_size * pp_group
            devices = tuple(
                i + zero_4_group for i in range(self.tensor_parallelism_size)
            )
            block_to_device_lookup.append(devices)

        config = LlamaModelConfig(
            hp=hp,
            device=self.torch_device,
            activation_dtype=self.activation_dtype,
            attention_dtype=self.attention_dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            tensor_parallelism_size=self.tensor_parallelism_size,
            pipeline_parallelism_size=self.pipeline_parallelism_size,
            block_seq_stride=self.block_seq_stride,
            attention_kernel=self.attention_kernel,
            use_hf=self.use_hf,
            block_to_device_lookup=block_to_device_lookup,
        )

        theta = dataset.root_theta

        model = PagedLlmModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

    def assemble_batch(self, token_batch: torch.tensor, devices) -> torch.tensor:

        token_batch, seq_lens_batch = self.generator.tokenizer.pad_tokens(
            token_ids=token_batch.tolist(),
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
        )

        logger.debug(f"{token_batch}")

        token_batch = torch.as_tensor(token_batch, device=self.torch_device)
        seq_lens_batch = torch.as_tensor(seq_lens_batch, device=self.torch_device)

        self.batch = self.generator.begin_batch(
            token_ids=token_batch,
            seq_lens=seq_lens_batch,
            page_cache_size=self.page_cache_size,
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
        logger.debug(f"Prefill input:")
        logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")

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
        logger.debug("Decode input:")
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

    @timeit
    def get_logits(self, skip_decode: bool) -> torch.Tensor:
        # Add context to improve perplexity by starting at 10th token
        self.start = 10
        shard_count = self.tensor_parallelism_size

        vm_instance = ireert.VmInstance()
        devices: list[iree.runtime.HalDevice] = get_iree_devices(
            device=self.iree_devices,
            device_count=self.pipeline_parallelism_size * shard_count,
            allow_repeating=True,
        )

        def run_iree_module(devices: list[iree.runtime.HalDevice]):
            hal_module = iree.runtime.create_hal_module(
                instance=vm_instance, devices=devices
            )
            weight_path = Path(self.weight_path_str)
            parameter_index = iree.runtime.ParameterIndex()
            if shard_count == 1:
                parameter_index.load(file_path=str(Path(weight_path)))
            else:
                for i in range(shard_count):
                    parameter_index.load(
                        file_path=str(
                            Path(weight_path).with_suffix(
                                f".rank{i}{weight_path.suffix}"
                            )
                        )
                    )

            parameter_provider = parameter_index.create_provider(scope="model")
            parameters_module = iree.runtime.create_io_parameters_module(
                vm_instance, parameter_provider
            )
            self.vm_module = iree.runtime.VmModule.mmap(
                vm_instance, str(self.output_vmfb)
            )
            self.vm_context = iree.runtime.VmContext(
                instance=vm_instance,
                modules=(hal_module, parameters_module, self.vm_module),
            )

            out_logits = []
            for i in tqdm(
                range(self.start, self.max_prompt_length - 1),
                mininterval=300,
                desc="eval: Calculating logits",
            ):
                logger.debug(f"Iteration: {i - self.start}")

                if skip_decode or len(out_logits) == 0:
                    token_batch = self.token_ids[:, : i + 1]

                    prefill_logits = self.prefill_vmfb(token_batch, i, devices).clone()
                    out_logits.append(prefill_logits[:, -1:, :])
                else:
                    token_batch = self.token_ids[:, i : i + 1]
                    decode_logits = self.decode_vmfb(token_batch, i, devices)
                    out_logits.append(decode_logits)

            out_logits = ops.cat(out_logits, dim=1)
            pad_logits_shape = self.token_ids.shape[1] - out_logits.shape[1]
            pad_logits = torch.zeros(
                out_logits.shape[0], pad_logits_shape, out_logits.shape[2]
            )

            self.cache_state = None  # Remove saved reference to iree.runtime.DeviceArray before leaving function
            return ops.cat((out_logits, pad_logits), 1).to(self.torch_device)

        return with_iree_device_context(run_iree_module, devices)

    def get_perplexity(
        self, test_prompts: list[str], skip_decode: bool
    ) -> dict[str, Any]:

        token_ids, seq_lens = self.generator.tokenizer.encode(
            test_prompts,
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
        )

        logger.debug(f" Prompts for Evaluation:")
        for idx, prompt in enumerate(test_prompts):
            logger.debug(
                f" Prompt {idx}: \nTokens: {prompt.encode()}\nToken ids: {token_ids[idx]}\n"
            )

        self.page_cache_size = (
            len(token_ids[0]) // self.generator.model.config.block_seq_stride
        ) * len(test_prompts) + 1

        self.max_prompt_length = max(seq_lens)

        self.token_ids = torch.as_tensor(token_ids, device=self.torch_device)

        out_logits = self.get_logits(skip_decode)

        logger.debug(f"Final Logits shape: {out_logits.shape}")
        logger.debug(f"Token ids shape: {self.token_ids.shape}")

        return compute_perplexity(self.token_ids, out_logits, self.start)


def run_perplexity_iree(
    args,
    dataset: Dataset,
    tokenizer: InferenceTokenizer,
    torch_device: torch.device,
    tensor_parallelism_size: int,
    pipeline_parallelism_size: int,
) -> dict[str, Any]:
    start = time.time()

    test_prompts = args.prompt_list or get_prompts(num_prompts=args.num_prompts)

    perplexity = PerplexityIree(
        torch_device=torch_device,
        iree_devices=args.iree_device,
        iree_hip_target=args.iree_hip_target,
        iree_hal_target_device=args.iree_hal_target_device,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelims_size=pipeline_parallelism_size,
        attention_kernel=args.attention_kernel,
        block_seq_stride=args.block_seq_stride,
        use_attention_mask=args.use_attention_mask,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        use_hf=args.use_hf,
        bs=len(test_prompts),
        weight_path_str=str(args.irpa_file),
    )

    perplexity.compile_model(
        output_mlir=args.output_mlir,
        output_config=args.output_config,
        output_vmfb=args.output_vmfb,
    )
    perplexity.load_model(dataset=dataset, tokenizer=tokenizer)
    perplexity_batch = perplexity.get_perplexity(
        test_prompts, skip_decode=args.skip_decode
    )

    end = time.time()
    total_time = round(end - start, 2)
    if total_time < 60:
        total_time = str(total_time) + " secs"
    else:
        total_time = str(round(total_time / 60, 2)) + " mins"
    logger.info(f" Total time taken: {total_time}")

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
    tokenizer = cli.get_tokenizer(args)

    logger.setLevel(args.loglevel)
    torch_device = torch.device(args.device) if args.device else None

    assert args.num_prompts or args.prompt_list, "Pass --num-prompts or --prompt-list"

    if args.output_mlir or args.output_config:
        assert (
            args.output_config is not None and args.output_mlir is not None
        ), "If using pre-exported mlir, both --mlir-path and --json-path must be passed"

    # Override flag if dataset disagrees
    tensor_parallelism_size = (
        dataset.properties["tensor_parallelism_size"]
        if "tensor_parallelism_size" in dataset.properties
        else args.tensor_parallelism_size
    )

    ppl = run_perplexity_iree(
        args,
        dataset=dataset,
        tokenizer=tokenizer,
        torch_device=torch_device,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
