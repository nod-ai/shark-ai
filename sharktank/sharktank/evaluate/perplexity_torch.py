# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import logging
import time
import json
import numpy as np
from tqdm import tqdm
import gc

import torch
from torch.nn import CrossEntropyLoss

from sharktank.layers import *
from sharktank.types import *

from sharktank.models.llm import *
from sharktank.models.llama.sharding import shard_theta

from sharktank.utils import cli
from sharktank.utils.load_llm import *
from sharktank.utils.evaluate import *


logger = logging.getLogger("eval")

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

    def __init__(self):
        pass

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

    def load_model(
        self,
        dataset: Dataset,
        tokenizer: InferenceTokenizer,
        tensor_parallelism_size: int,
        device: torch.device,
        activation_dtype: torch.dtype,
        attention_dtype: torch.dtype,
        kv_cache_dtype: torch.dtype,
        attention_kernel: str,
        block_seq_stride: int,
        use_hf: bool,
        fake_quant: bool,
    ):

        config = LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
            device=device,
            activation_dtype=activation_dtype,
            attention_dtype=attention_dtype,
            kv_cache_dtype=kv_cache_dtype,
            tensor_parallelism_size=tensor_parallelism_size,
            block_seq_stride=block_seq_stride,
            attention_kernel=attention_kernel,
            use_hf=use_hf,
            fake_quant=fake_quant,
        )

        self.device = device

        if config.tensor_parallelism_size > 1:
            dataset.root_theta = shard_theta(dataset.root_theta, config)

        theta = dataset.root_theta

        model = PagedLlmModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

    def assemble_batch(self, token_batch: torch.tensor) -> torch.tensor:

        token_batch, seq_lens_batch = self.generator.tokenizer.pad_tokens(
            token_ids=token_batch.tolist(),
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
        )

        logger.debug(f"{token_batch}")

        token_batch = torch.as_tensor(token_batch, device=self.device)
        seq_lens_batch = torch.as_tensor(seq_lens_batch, device=self.device)

        self.batch = self.generator.begin_batch(
            token_ids=token_batch,
            seq_lens=seq_lens_batch,
            page_cache_size=self.page_cache_size,
        )

        return token_batch

    def get_logits(self, skip_decode: bool) -> torch.tensor:

        is_first_token = True
        # Add context to improve perplexity by starting at 10th token
        self.start = 10
        out_logits = []
        for i in range(self.start, self.max_prompt_length - 1):
            logger.debug(f"Iteration: {i}")

            if is_first_token:

                token_batch = self.token_ids[:, : i + 1]
                logger.debug(f"Prefill input:")
                logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")

                token_batch = self.assemble_batch(token_batch)

                self.batch.prefill()
                out_logits.append(self.batch.prefill_logits[:, 0:1, :])

                self.print_token_comparison(i)

                if not skip_decode:
                    is_first_token = False

            else:
                token_batch = self.token_ids[:, i : i + 1]

                logger.debug("Decode input:")
                logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")
                logger.debug(f"{token_batch.tolist()}")

                self.batch.decode(token_batch=token_batch)
                out_logits.append(self.batch.decode_logits)

                self.print_token_comparison(i)

        out_logits = torch.cat(out_logits, dim=1)

        pad_logits_shape = self.token_ids.shape[1] - out_logits.shape[1]

        pad_logits = torch.zeros(
            out_logits.shape[0],
            pad_logits_shape,
            out_logits.shape[2],
            device=self.device,
        )

        out_logits = torch.cat((out_logits, pad_logits), dim=1).to(self.device)

        return out_logits

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

        self.token_ids = torch.tensor(token_ids, device=self.device)

        out_logits = self.get_logits(skip_decode)

        logger.debug(f"Final Logits shape: {out_logits.shape}")
        logger.debug(f"Token ids shape: {self.token_ids.shape}")

        return compute_perplexity(self.token_ids, out_logits, self.start)


def run_perplexity_torch(
    args,
    dataset,
    tokenizer,
    device,
    tensor_parallelism_size,
    model_file,
):

    start = time.time()

    test_prompts = args.prompt_list or get_prompts(num_prompts=args.num_prompts)

    bs = 32

    input_prompts = [
        test_prompts[idx : idx + bs] for idx in range(0, len(test_prompts), bs)
    ]

    perplexity_batch = []
    for p in tqdm(
        input_prompts,
        desc=f"eval: Calculating logits for {model_file.name}",
    ):
        perplexity_batch.extend(
            perplexity_torch(
                dataset=dataset,
                tokenizer=tokenizer,
                device=device,
                tensor_parallelism_size=tensor_parallelism_size,
                attention_kernel=args.attention_kernel,
                block_seq_stride=args.block_seq_stride,
                prompts=p,
                activation_dtype=args.activation_dtype,
                attention_dtype=args.attention_dtype,
                kv_cache_dtype=args.kv_cache_dtype,
                use_hf=args.use_hf,
                fake_quant=args.fake_quant,
                skip_decode=args.skip_decode,
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
    tensor_parallelism_size,
    attention_kernel,
    block_seq_stride,
    prompts,
    activation_dtype,
    attention_dtype,
    kv_cache_dtype,
    use_hf,
    fake_quant,
    skip_decode,
):

    perplexity = PerplexityTorch()

    perplexity.load_model(
        dataset=dataset,
        tokenizer=tokenizer,
        tensor_parallelism_size=tensor_parallelism_size,
        device=device,
        activation_dtype=activation_dtype,
        attention_dtype=attention_dtype,
        kv_cache_dtype=kv_cache_dtype,
        attention_kernel=attention_kernel,
        block_seq_stride=block_seq_stride,
        use_hf=use_hf,
        fake_quant=fake_quant,
    )

    ppl = perplexity.get_perplexity(prompts, skip_decode=skip_decode)

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
    tokenizer = cli.get_tokenizer(args)

    logger.setLevel(args.loglevel)
    device = torch.device(args.device) if args.device else None
    model_file = args.gguf_file or args.irpa_file

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
        model_file=model_file,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")

    gc.collect()
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
