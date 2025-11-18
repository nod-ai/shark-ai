# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
import sys
import logging
import time
from datetime import timedelta
import json
import numpy as np
from tqdm import tqdm
import gc

import torch

from amdsharktank.layers import *
from amdsharktank.types import *

from amdsharktank.models.llm import *
from amdsharktank.types.pipelining import pipeline_parallelize_llm_theta

from amdsharktank.utils import cli
from amdsharktank.utils.load_llm import *
from amdsharktank.utils.logging import get_logger
from amdsharktank.utils.evaluate import *
import amdsharktank.ops as ops

logger = get_logger(__name__)

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)

__all__ = ["PerplexityTorch", "run_perplexity_torch"]


class PerplexityTorch:
    """
    Perplexity (PPL) is one of the most common metrics for evaluating language models.
    It is defined as the exponentiated average negative log-likelihood of a sequence,
    calculated with exponent base `e`.

    For more information, see https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        prefill_length: int | None = None,
        use_toy_model: bool = False,
    ):
        assert prefill_length is None or prefill_length >= 1
        self.prefill_length = prefill_length
        self.use_toy_model = use_toy_model

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
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

    @timeit
    def load_model(
        self,
        dataset: Dataset,
        tensor_parallelism_size: int,
        pipeline_parallelism_size: int,
        device: torch.device,
        activation_dtype: torch.dtype,
        attention_dtype: torch.dtype,
        kv_cache_dtype: torch.dtype,
        attention_kernel: str,
        block_seq_stride: int,
        use_hf: bool,
        fake_quant: bool,
        tokenizer: Optional[InferenceTokenizer] = None,
    ):

        config = configs.LlamaModelConfig.from_dataset(
            dataset,
            device=device,
            activation_dtype=activation_dtype,
            attention_dtype=attention_dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_seq_stride=block_seq_stride,
            attention_kernel=attention_kernel,
            fake_quant=fake_quant,
        )

        hp = config.hp
        hp.rope_interleave_emb = not use_hf
        config.parallelism_config = ParallelismConfig.default_config(
            block_count=hp.block_count,
            pp=pipeline_parallelism_size,
            tp=tensor_parallelism_size,
        )

        pipeline_parallelize_llm_theta(dataset.root_theta, config.parallelism_config)

        self.device = device

        theta = dataset.root_theta

        model = PagedLlmModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

    def assemble_batch(self, token_batch: torch.tensor) -> torch.tensor:

        token_batch, seq_lens_batch = pad_tokens(
            token_ids=token_batch.tolist(),
            pad_to_multiple_of=self.generator.model.config.block_seq_stride,
        )

        logger.debug(f"{token_batch}")

        token_batch = torch.as_tensor(token_batch, device=self.device)
        seq_lens_batch = torch.as_tensor(seq_lens_batch, device=self.device)

        self.batch = self.generator.begin_batch(
            token_ids=token_batch,
            seq_lens=seq_lens_batch,
            page_cache_size=self.page_cache_size,
            max_decode_steps=self.last_token_index - self.prefill_length - 1,
        )

        return token_batch

    def get_logits(self, skip_decode: bool) -> torch.tensor:

        is_first_token = True
        out_logits = []
        self.last_token_index = self.max_prompt_length
        context_length = self.generator.model.config.hp.context_length
        if self.last_token_index > context_length:
            logger.warning(
                f"Last token {self.last_token_index} exceeds context length {context_length}. "
                "Limiting tokens to context length."
            )
            self.last_token_index = context_length
        for i in range(self.prefill_length - 1, self.last_token_index - 1):
            logger.debug(f"Iteration: {i - self.prefill_length + 1}")

            if is_first_token:

                token_batch = self.token_ids[:, : i + 1]
                if not self.use_toy_model:
                    logger.debug(
                        f"Prefill input:\n{self.generator.tokenizer.decode(token_batch)}"
                    )

                token_batch = self.assemble_batch(token_batch)

                self.batch.prefill()

                last_logits_indices = torch.minimum(self.seq_lens - 1, torch.tensor(i))
                last_logits_indices = torch.maximum(
                    last_logits_indices, torch.tensor(0)
                )
                batch_indices = torch.arange(len(self.seq_lens))
                last_real_prefill_logits = self.batch.prefill_logits[
                    batch_indices, last_logits_indices, :
                ].unsqueeze(1)
                out_logits.append(last_real_prefill_logits)

                self.print_token_comparison(i)

                if not skip_decode:
                    is_first_token = False

            else:
                token_batch = self.token_ids[:, i : i + 1]

                logger.debug(f"Decode input:")
                if not self.use_toy_model:
                    logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")
                logger.debug(f"{token_batch.tolist()}")

                self.batch.decode(token_batch=token_batch)
                out_logits.append(self.batch.decode_logits)

                self.print_token_comparison(i)

        out_logits = ops.cat(out_logits, dim=1)

        pad_logits_shape = self.token_ids.shape[1] - out_logits.shape[1]

        pad_logits = torch.zeros(
            out_logits.shape[0],
            pad_logits_shape,
            out_logits.shape[2],
            device=self.device,
        )

        return ops.cat(
            (
                pad_logits[:, : self.prefill_length],
                out_logits,
                pad_logits[:, self.prefill_length :],
            ),
            dim=1,
        ).to(self.device)

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
        self.seq_lens = torch.tensor(self.seq_lens, device=self.device)

        self.token_ids = torch.tensor(self.token_ids, device=self.device)

        out_logits = self.get_logits(skip_decode)

        logger.debug(f"Final Logits shape: {out_logits.shape}")
        logger.debug(f"Token ids shape: {self.token_ids.shape}")

        return compute_perplexity(
            self.token_ids, out_logits, self.prefill_length - 1, self.last_token_index
        )


def run_perplexity_torch(
    args,
    dataset,
    tokenizer,
    device: torch.device | None,
    tensor_parallelism_size: int,
    pipeline_parallelism_size: int,
):

    start = time.time()

    token_ids = None
    test_prompts = None

    if args.use_toy_model:
        token_ids = get_token_ids()
        bs = len(token_ids)
        input_prompts = [
            token_ids[idx : idx + bs] for idx in range(0, len(token_ids), bs)
        ]
    else:
        test_prompts = args.prompt_list or get_prompts(num_prompts=args.num_prompts)
        bs = len(test_prompts)
        input_prompts = [
            test_prompts[idx : idx + bs] for idx in range(0, len(test_prompts), bs)
        ]

    model_file = args.gguf_file or args.irpa_file
    perplexity_batch = []
    for p in tqdm(
        input_prompts,
        desc=f"eval_torch: Calculating logits for {model_file.name}",
    ):
        perplexity_batch.extend(
            perplexity_torch(
                dataset=dataset,
                tokenizer=tokenizer,
                device=device,
                tensor_parallelism_size=tensor_parallelism_size,
                pipeline_parallelism_size=pipeline_parallelism_size,
                attention_kernel=args.attention_kernel,
                block_seq_stride=args.block_seq_stride,
                prompts=p,
                token_ids=token_ids,
                activation_dtype=args.activation_dtype,
                attention_dtype=args.attention_dtype,
                kv_cache_dtype=args.kv_cache_dtype,
                use_hf=args.use_hf,
                fake_quant=args.fake_quant,
                skip_decode=args.skip_decode,
                use_toy_model=args.use_toy_model,
            )
        )

        gc.collect()

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


def perplexity_torch(
    dataset,
    tokenizer,
    device,
    tensor_parallelism_size: int,
    pipeline_parallelism_size: int,
    attention_kernel,
    block_seq_stride,
    prompts,
    token_ids,
    activation_dtype,
    attention_dtype,
    kv_cache_dtype,
    use_hf,
    fake_quant,
    skip_decode,
    use_toy_model,
):
    perplexity = PerplexityTorch(use_toy_model=use_toy_model)

    perplexity.load_model(
        dataset=dataset,
        tokenizer=tokenizer,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism_size=pipeline_parallelism_size,
        device=device,
        activation_dtype=activation_dtype,
        attention_dtype=attention_dtype,
        kv_cache_dtype=kv_cache_dtype,
        attention_kernel=attention_kernel,
        block_seq_stride=block_seq_stride,
        use_hf=use_hf,
        fake_quant=fake_quant,
    )

    ppl = perplexity.get_perplexity(
        test_prompts=prompts,
        token_ids=token_ids,
        skip_decode=skip_decode,
    )

    return ppl


def main(argv):
    parser = cli.create_parser()

    cli.add_evaluate_options(parser)
    cli.add_model_options(parser)
    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    cli.add_quantization_options(parser)
    cli.add_log_options(parser)

    args = cli.parse(parser, args=argv)

    dataset = cli.get_input_dataset(args)
    tokenizer = None
    if not args.use_toy_model:
        tokenizer = cli.get_tokenizer(args)

    logger.setLevel(args.loglevel)
    device = torch.device(args.device) if args.device else None

    assert args.num_prompts or args.prompt_list, "Pass --num-prompts or --prompt-list"

    # Override flag if dataset disagrees
    tensor_parallelism_size = (
        dataset.properties["tensor_parallelism_size"]
        if "tensor_parallelism_size" in dataset.properties
        else args.tensor_parallelism_size
    )

    ppl = run_perplexity_torch(
        args,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")

    gc.collect()
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
