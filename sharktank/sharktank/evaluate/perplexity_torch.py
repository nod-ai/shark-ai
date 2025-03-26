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

from sharktank.models.llama.llama import *
from sharktank.models.mixtral.mixtral import *
from sharktank.models.grok.grok import *

from ..models.llama.sharding import shard_theta

from sharktank.utils import cli
from sharktank.utils.load_llm import *
from sharktank.utils.evaluate import *


log_levels = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
logger = logging.getLogger("eval")

logger.setLevel(log_levels["info"])

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)

__all__ = ["Perplexity_torch", "run_perplexity_torch"]


class Perplexity_torch:
    """
    Perplexity (PPL) is one of the most common metrics for evaluating language models.
    It is defined as the exponentiated average negative log-likelihood of a sequence,
    calculated with exponent base `e`.

    For more information, see https://huggingface.co/docs/transformers/perplexity
    """

    def __init__(
        self,
        device,
        use_hf,
        fake_quant,
        test_prompts,
        activation_dtype=torch.float32,
        attention_dtype=torch.float32,
        kv_cache_dtype=torch.float32,
    ):
        self.device = device
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.use_hf = use_hf
        self.fake_quant = fake_quant
        self.test_prompts = test_prompts
        self.bs = len(self.test_prompts)

    def print_token_comparison(self, i):
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

    def load_model(self, dataset, tokenizer, tensor_parallelism_size, attention_kernel):

        self.config = LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
            device=self.device,
            activation_dtype=self.activation_dtype,
            attention_dtype=self.attention_dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            tensor_parallelism_size=tensor_parallelism_size,
            use_hf=self.use_hf,
            fake_quant=self.fake_quant,
        )

        if self.config.tensor_parallelism_size > 1:
            dataset.root_theta = shard_theta(dataset.root_theta, self.config)

        theta = dataset.root_theta

        if self.config.hp.expert_count:
            if self.config.hp.model_arch == "grok":
                model = PagedGrokModelV1(theta, self.config)
            else:
                model = PagedMixtralModelV1(theta, self.config)
        else:
            model = PagedLlamaModelV1(theta, self.config)

        self.generator = TorchGenerator(model, tokenizer)

    def get_logits(self, page_cache_size):

        is_first_token = True
        self.start = 0
        self.out_logits = []
        for i in range(self.start, self.max_prompt_length - 1):
            logger.debug(f"Iteration: {i}")

            if is_first_token:

                token_batch = self.token_ids[:, : i + 1]
                logger.debug(f"Prefill:")

                logger.debug("Input:")
                logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")

                token_batch, seq_lens_batch = self.generator.tokenizer.pad_tokens(
                    token_ids=token_batch.tolist(),
                    pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
                )

                logger.debug(f"{token_batch}")

                token_batch = torch.tensor(token_batch, device=self.device)
                seq_lens_batch = torch.tensor(seq_lens_batch, device=self.device)

                self.batch = self.generator.begin_eval_batch(
                    token_batch=token_batch,
                    seq_lens_batch=seq_lens_batch,
                    bs=self.bs,
                    page_cache_size=page_cache_size,
                )

                self.batch.prefill()
                self.out_logits.append(self.batch.prefill_logits[:, 0:1, :])

                is_first_token = False

                self.print_token_comparison(i)

            else:
                token_batch = self.token_ids[:, i : i + 1]

                logger.debug("Decode:")

                logger.debug("Input:")
                logger.debug(f"{self.generator.tokenizer.decode(token_batch)}")
                logger.debug(f"{token_batch.tolist()}")

                self.batch.decode(token_batch=token_batch)
                self.out_logits.append(self.batch.decode_logits)

                self.print_token_comparison(i)

        self.out_logits = torch.cat(self.out_logits, 1)

        pad_logits_shape = self.token_ids.shape[1] - self.out_logits.shape[1]

        self.pad_logits = torch.zeros(
            self.out_logits.shape[0],
            pad_logits_shape,
            self.out_logits.shape[2],
            device=self.device,
        )

        self.out_logits = torch.cat((self.out_logits, self.pad_logits), 1)

    def compute_perplexity(self):
        loss_fct = CrossEntropyLoss(reduction="none")

        ## perplexity = e ^ (sum(losses) / num_tokenized_tokens)

        crossentropy_loss = (
            loss_fct(self.out_logits.transpose(1, 2), self.token_ids)
            * self.attention_mask
        ).sum(1)
        crossentropy_loss = torch.tensor(crossentropy_loss.tolist(), device=self.device)
        perplexity_batch = torch.exp(
            crossentropy_loss / self.attention_mask.sum(1)
        ).tolist()

        perplexity_batch = [round(ppl, 6) for ppl in perplexity_batch]

        return perplexity_batch

    def get_perplexity(self):

        token_ids, seq_lens = self.generator.tokenizer.encode(
            self.test_prompts,
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
        )

        logger.debug(f" Prompts for Evaluation:")
        for idx, prompt in enumerate(self.test_prompts):
            logger.debug(
                f" Prompt {idx}: \nTokens: {prompt.encode()}\nToken ids: {token_ids[idx]}\n"
            )

        self.page_cache_size = (
            len(token_ids[0]) // self.config.block_seq_stride
        ) * self.bs + 1

        self.max_prompt_length = max(seq_lens)

        self.token_ids = torch.tensor(token_ids, device=self.device)
        self.attention_mask = (
            (self.token_ids != 0).int().detach().clone().to(self.device)
        )

        self.get_logits(page_cache_size=self.page_cache_size)

        self.out_logits = self.out_logits[..., :-1, :].contiguous()
        self.token_ids = self.token_ids[..., 1:].contiguous()
        self.attention_mask = self.attention_mask[..., 1:].contiguous()

        logger.debug(f"Final Logits shape: {self.out_logits.shape}")
        logger.debug(f"Token ids: {self.token_ids}, \n{self.token_ids.shape}")
        logger.debug(
            f"Mask shape: {self.attention_mask}, \n{self.attention_mask.shape}"
        )

        assert self.token_ids.shape == self.out_logits.shape[0:2]

        return self.compute_perplexity()


def run_perplexity_torch(
    dataset,
    tokenizer,
    device,
    tensor_parallelism_size,
    attention_kernel,
    num_prompts,
    activation_dtype,
    attention_dtype,
    kv_cache_dtype,
    use_hf,
    fake_quant,
    model_file,
):

    prompts = get_prompts(num_prompts=num_prompts)

    logger.info(f" Batch size: {len(prompts)}")
    start = time.time()

    bs = 32

    input_prompts = [prompts[idx : idx + bs] for idx in range(0, len(prompts), bs)]
    perplexity_batch = []
    for p in tqdm(
        input_prompts,
        desc=f"eval: Calculating logits for {model_file.name}",
    ):
        perplexity_batch.extend(
            perplexity_torch(
                dataset,
                tokenizer,
                device,
                tensor_parallelism_size,
                attention_kernel,
                p,
                activation_dtype,
                attention_dtype,
                kv_cache_dtype,
                use_hf,
                fake_quant,
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
    prompts,
    activation_dtype,
    attention_dtype,
    kv_cache_dtype,
    use_hf,
    fake_quant,
):

    perplexity = Perplexity_torch(
        device=device,
        test_prompts=prompts,
        activation_dtype=activation_dtype,
        attention_dtype=attention_dtype,
        kv_cache_dtype=kv_cache_dtype,
        fake_quant=fake_quant,
        use_hf=use_hf,
    )

    perplexity.load_model(dataset, tokenizer, tensor_parallelism_size, attention_kernel)
    ppl = perplexity.get_perplexity()
    return ppl


def main(argv):
    parser = cli.create_parser()

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts for perplexity test",
    )

    cli.add_model_options(parser)
    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    cli.add_quantization_options(parser)

    args = cli.parse(parser, args=argv)

    device = torch.device(args.device) if args.device else None
    model_file = args.gguf_file or args.irpa_file
    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)

    # Override flag if dataset disagrees
    tensor_parallelism_size = (
        dataset.properties["tensor_parallelism_size"]
        if "tensor_parallelism_size" in dataset.properties
        else args.tensor_parallelism_size
    )

    ppl = run_perplexity_torch(
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        tensor_parallelism_size=tensor_parallelism_size,
        attention_kernel=args.attention_kernel,
        num_prompts=args.num_prompts,
        attention_dtype=args.attention_dtype,
        activation_dtype=args.activation_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        use_hf=args.use_hf,
        fake_quant=args.fake_quant,
        model_file=model_file,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")

    gc.collect()
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
