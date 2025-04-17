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

from sharktank.models.llm import *

from sharktank.layers import *
from sharktank.types import *

from sharktank.utils import cli
from sharktank.utils.vmfb_runner import *
from sharktank.utils.load_llm import *
from sharktank.utils.create_cache import *
from sharktank.utils.export_artifacts import *
from sharktank.utils.evaluate import *
from sharktank.utils.iree import (
    iree_to_torch,
    with_iree_device_context,
    torch_tensor_to_device_array,
)

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
        iree_device,
        iree_hip_target,
        iree_hal_target_device,
        bs,
        tensor_parallelism_size,
        attention_kernel,
        block_seq_stride,
        activation_dtype,
        attention_dtype,
        kv_cache_dtype,
        use_attention_mask,
        use_hf,
    ):
        self.torch_device = torch_device
        self.iree_device = iree_device
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_device = iree_hal_target_device
        self.bs = bs
        self.tensor_parallelism_size = tensor_parallelism_size
        self.attention_kernel = attention_kernel
        self.block_seq_stride = block_seq_stride
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.use_attention_mask = use_attention_mask
        self.use_hf = use_hf

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
        weight_path_str: str,
        output_mlir: str,
        output_config: str,
        output_vmfb: str,
    ):
        self.weight_path_str = weight_path_str

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

        config = LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
            device=self.torch_device,
            activation_dtype=self.activation_dtype,
            attention_dtype=self.attention_dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            tensor_parallelism_size=self.tensor_parallelism_size,
            block_seq_stride=self.block_seq_stride,
            attention_kernel=self.attention_kernel,
            use_hf=self.use_hf,
        )

        theta = dataset.root_theta

        model = PagedLlmModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

        self.runner = vmfbRunner(
            device=self.iree_device,
            vmfb_path=self.output_vmfb,
            external_weight_path=self.weight_path_str,
        )

        self.haldevice = self.runner.config.device

    def assemble_batch(self, token_batch: torch.tensor) -> torch.tensor:

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

        self.cache_state = torch_tensor_to_device_array(
            self.batch.cache_state[0], self.haldevice
        )
        return token_batch

    def prefill_vmfb(self, token_batch: torch.tensor, i: int) -> torch.tensor:

        logger.debug(f"Prefill input:")
        logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")

        token_batch = self.assemble_batch(token_batch)

        seq_block_ids = self.batch.pad_block_ids()
        prefill_logits = self.runner.ctx.modules.module[f"prefill_bs{self.bs}"](
            token_batch,
            self.batch.seq_lens,
            seq_block_ids,
            self.cache_state,
        )
        prefill_logits = torch.as_tensor(iree_to_torch(prefill_logits)[0][:, :, :])

        tokens = torch.as_tensor(
            self.generator.model.extract_tokens_from_logits(
                prefill_logits, self.batch.seq_lens
            )
        ).unsqueeze(1)
        self.batch.add_result_token(tokens)

        self.print_token_comparison(i)
        return prefill_logits

    def decode_vmfb(self, token_batch: torch.tensor, i: int) -> torch.tensor:
        logger.debug("Decode input:")
        logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")
        logger.debug(f"{token_batch.tolist()}")

        start_positions = self.batch.seq_lens.clone()
        self.batch.seq_lens.add_(1)
        self.batch.allocate_seq_block_ids()
        seq_block_ids = self.batch.pad_block_ids()

        decode_logits = self.runner.ctx.modules.module[f"decode_bs{self.bs}"](
            token_batch,
            self.batch.seq_lens,
            start_positions,
            seq_block_ids,
            self.cache_state,
        )
        decode_logits = torch.as_tensor(iree_to_torch(decode_logits)[0][:, :, :])

        tokens = torch.as_tensor(
            self.generator.model.extract_tokens_from_logits(
                decode_logits, [1] * self.bs
            ),
            device=self.generator.model.device,
        ).unsqueeze(1)
        self.batch.add_result_token(tokens)
        self.print_token_comparison(i)
        return decode_logits

    def get_logits(self, skip_decode: bool) -> torch.tensor:
        # Add context to improve perplexity by starting at 10th token
        self.start = 10
        self.out_logits = []

        def run_iree_module(iree_devices: list[ireert.HalDevice]):
            iter = 0
            is_first_token = True
            for i in tqdm(
                range(self.start, self.max_prompt_length - 1),
                mininterval=300,
                desc="eval: Calculating logits",
            ):
                logger.debug(f"Iteration: {iter}")

                if is_first_token:

                    token_batch = self.token_ids[:, : i + 1]

                    prefill_logits = self.prefill_vmfb(token_batch, i)

                    self.out_logits.append(prefill_logits[:, -1:, :])

                    if not skip_decode:
                        is_first_token = False

                else:
                    token_batch = self.token_ids[:, i : i + 1]
                    decode_logits = self.decode_vmfb(token_batch, i)
                    self.out_logits.append(decode_logits)

                iter += 1

            out_logits = torch.cat(self.out_logits, dim=1)

            pad_logits_shape = self.token_ids.shape[1] - out_logits.shape[1]

            pad_logits = torch.zeros(
                out_logits.shape[0], pad_logits_shape, out_logits.shape[2]
            )

            out_logits = torch.cat((out_logits, pad_logits), 1).to(self.torch_device)

            return out_logits

        return with_iree_device_context(run_iree_module, [self.runner.config.device])

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
) -> dict[str, Any]:

    start = time.time()

    test_prompts = args.prompt_list or get_prompts(num_prompts=args.num_prompts)

    perplexity = PerplexityIree(
        torch_device=torch_device,
        iree_device=args.iree_device,
        iree_hip_target=args.iree_hip_target,
        iree_hal_target_device=args.iree_hal_target_device,
        tensor_parallelism_size=tensor_parallelism_size,
        attention_kernel=args.attention_kernel,
        block_seq_stride=args.block_seq_stride,
        use_attention_mask=args.use_attention_mask,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        use_hf=args.use_hf,
        bs=len(test_prompts),
    )

    perplexity.compile_model(
        weight_path_str=str(args.irpa_file),
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
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
