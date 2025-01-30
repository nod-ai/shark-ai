# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import os
import logging
import json
import time
import random
import re
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
import uuid
import requests

import numpy as np

from datasets import load_dataset

import torch
from torch.nn import CrossEntropyLoss

from integration_tests.llm.utils import (
    export_paged_llm_v1,
    compile_model,
    find_available_port,
    start_llm_server,
    start_log_group,
    end_log_group,
)

log_levels = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
logger = logging.getLogger("eval")

logger.setLevel(log_levels["info"])

logger.root.handlers[0].setFormatter(
    logging.Formatter(fmt="\n%(levelname)s:%(name)-8s %(message)s")
)

__all__ = ["Perplexity", "run_perplexity"]


class Perplexity:
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
        kv_cache_type,
        tensor_parallelism_size,
        attention_kernel,
        block_seq_stride,
    ):
        self.torch_device = torch_device
        self.iree_device = iree_device
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_device = iree_hal_target_device
        self.kv_cache_type = kv_cache_type
        self.block_seq_stride = block_seq_stride
        self.activation_dtype = torch.float16
        self.attention_dtype = torch.float16
        self.tensor_parallelism_size = tensor_parallelism_size
        self.attention_kernel = attention_kernel

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            total_seconds = end - start
            time_taken = abs(timedelta(seconds=total_seconds))
            hours, minutes, seconds = re.split(":", str(time_taken))

            if total_seconds < 1:
                time_taken = f" {round(total_seconds * 1000, 3)} ms"
            elif total_seconds < 60:
                time_taken = "{:.2f} secs".format(round(float(total_seconds), 2))
            else:
                time_taken = "{:02d} hrs : {:02d} mins : {:.2f} secs".format(
                    int(hours), int(minutes), round(float(seconds), 2)
                )

            func_name = func.__name__
            if func_name == "get_perplexity":
                func_name = f"Calculate perplexity"
            elif func_name == "compile_model":
                func_name = f"Export & compile"
            logger.info(f" {func_name}: {time_taken}")
            return result

        return wrapper

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

    @timeit
    def compile_model(self, weight_path_str):
        # self.sharktank_dir = str(
        #     Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent
        # )
        # self.weight_path_str = weight_path_str

        # (
        #     self.mlir_path,
        #     self.config_path,
        #     self.vmfb_path,
        #     self.edited_config_path,
        # ) = get_artifacts(source_dir=self.sharktank_dir, irpa_path=self.weight_path_str)

        # Export model
        gpu_settings = {
            "device_flags": [
                f"-iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
            ],
            "device": "hip",
        }

        batch_sizes = [self.bs]

        export_paged_llm_v1(
            self.mlir_path, self.config_path, self.weight_path_str, batch_sizes
        )

        # Compile model
        compile_model(self.mlir_path, self.vmfb_path, device_settings=gpu_settings)

        # Write config - not needed anymore
        # prefix_sharing_algorithm = "none"  # trie
        # config = {
        #     "module_name": "module",
        #     "module_abi_version": 1,
        #     "max_seq_len": 2048,
        #     "attn_head_count": 32,
        #     "attn_head_dim": 100,
        #     "prefill_batch_sizes": batch_sizes,
        #     "decode_batch_sizes": batch_sizes,
        #     "transformer_block_count": 26,
        #     "paged_kv_cache": {
        #         "block_seq_stride": 16,
        #         "device_block_count": 256,
        #         "prefix_sharing_algorithm": prefix_sharing_algorithm,
        #     },
        # }
        # logger.info(f"Saving edited config to: {self.edited_config_path}\n")
        # logger.info(f"Config: {json.dumps(config, indent=2)}")
        # with open(self.edited_config_path, "w") as f:
        #     json.dump(config, f)

    @timeit
    def get_prompts(self, num_prompts):
        test_prompts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[
            "text"
        ]
        num_test_prompts = 219

        random.seed(0)
        test_prompts = random.sample(test_prompts, num_test_prompts)

        # Ignore prompts that are: empty, less than 20 tokens or a title.
        test_prompts = [
            s.replace("\n", "").rstrip()
            for s in test_prompts
            if s != "" and len(s.split()) >= 20 and s.count("=") < 2
        ][0:num_prompts]

        self.test_prompts = test_prompts

        self.bs = len(test_prompts)

        logger.info(f" Batch size: {self.bs}")

    @timeit
    def generate(self, prompt: str | list[int], port: int, input_ids=False) -> str:
        """Helper method to make generation request to server.

        Args:
            prompt: Input text prompt
            port: Server port number

        Returns:
            Generated text response

        Raises:
            requests.exceptions.RequestException: If request fails
            AccuracyValidationException: If response format is invalid
        """
        payload = {
            "sampling_params": {"max_completion_tokens": 15, "temperature": 0.7},
            "rid": uuid.uuid4().hex,
            "stream": False,
            "return_logprob": True,  # dummy param in /home/aramalin/shark-ai/shortfin/python/shortfin_apps/llm/components/io_struct.py needs to be plumbed thro' to return logits
        }
        if input_ids:
            payload["input_ids"] = prompt
        else:
            payload["text"] = prompt
        response = requests.post(
            f"http://localhost:{port}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,  # Add reasonable timeout
        )
        response.raise_for_status()

        # TODO: Parse response for logits
        logits = response.text

        return logits

    @timeit
    def get_logits(self, page_cache_size):

        # TODO:
        """
        GenerateReqInput /io struct- input format-add groundtruth prompt as new arg (Done)
        generate.py - prefill decode reqs created-use the groundtruth prompt tokens instead of predicted toks in decode loop (follow TODOs)
        from integration_tests.llm.utils import (start_llm_server) = starts the llm server (ref: shark-ai/app_tests/benchmark_tests/llm/sglang_benchmarks/shortfin_benchmark_test.py)
        Optional, if perplexity_tests/llm/generate.py logits don't work: Plumb thro' return_logprob param to actually return logits, service.py sends logits - https://github.com/nod-ai/shark-ai/blob/04d383b5a67de031bf6e8626a84d030c346792eb/shortfin/python/shortfin_apps/llm/components/service.py#L479
        """
        # TODO: run do_generate and get logits here

        is_first_token = True
        start = 0
        for i in tqdm(
            range(start, self.max_prompt_length - 1),
            mininterval=300,
            desc="eval: Calculating logits",
        ):
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

                token_batch = torch.tensor(token_batch, device=self.torch_device)
                self.seq_lens_batch = torch.tensor(
                    seq_lens_batch, device=self.torch_device
                )

                self.batch = self.generator.begin_eval_batch(
                    token_batch=token_batch,
                    seq_lens_batch=self.seq_lens_batch,
                    bs=self.bs,
                    page_cache_size=page_cache_size,
                )

                self.cache_state = ireert.asdevicearray(
                    self.haldevice, self.batch.cache_state[0].to("cpu").numpy()
                )

                prefill_logits = self.prefill_vmfb(token_batch, i)
                self.out_logits = prefill_logits[:, -1:, :]

                is_first_token = False

            else:
                token_batch = self.token_ids[:, i : i + 1]

                decode_logits = self.decode_vmfb(token_batch, i)
                self.out_logits = torch.cat((self.out_logits, decode_logits), 1)

        pad_logits_shape = self.token_ids.shape[1] - self.out_logits.shape[1]

        self.pad_logits = torch.zeros(
            self.out_logits.shape[0], pad_logits_shape, self.out_logits.shape[2]
        )

        self.out_logits = torch.cat((self.out_logits, self.pad_logits), 1).to(
            self.torch_device
        )

    @timeit
    def compute_perplexity(self):
        loss_fct = CrossEntropyLoss(reduction="none")

        ## perplexity = e ^ (sum(losses) / num_tokenized_tokens)
        crossentropy_loss = (
            loss_fct(self.out_logits.transpose(1, 2), self.token_ids)
            * self.attention_mask
        ).sum(1)
        crossentropy_loss = torch.tensor(crossentropy_loss.tolist())
        perplexity_batch = torch.exp(
            crossentropy_loss / self.attention_mask.sum(1)
        ).tolist()

        perplexity_batch = [round(ppl, 6) for ppl in perplexity_batch]

        return {
            "perplexities": perplexity_batch,
            "mean_perplexity": round(np.mean(perplexity_batch), 6),
        }

    @timeit
    def get_perplexity(self):
        # TODO: replace tokenizer with shortfin one

        token_ids, seq_lens = self.generator.tokenizer.encode(
            self.test_prompts,
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
        )

        self.page_cache_size = (
            len(token_ids[0]) // self.config.block_seq_stride
        ) * self.bs + 1

        logger.debug(f" Prompts for Evaluation:")
        for idx, prompt in enumerate(self.test_prompts):
            logger.debug(
                f" Prompt {idx}: \nTokens: {prompt.encode()}\nToken ids: {token_ids[idx]}\n"
            )

        self.max_prompt_length = max(seq_lens)

        self.token_ids = torch.tensor(token_ids, device=self.torch_device)
        self.attention_mask = (
            (self.token_ids != 0).int().detach().clone().to(self.torch_device)
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


def run_perplexity(
    weight_path,
    weight_path_str,
    tokenizer,
    torch_device,
    iree_device,
    iree_hip_target,
    iree_hal_target_device,
    kv_cache_type,
    tensor_parallelism_size,
    attention_kernel,
    num_prompts,
    block_seq_stride,
):
    start = time.time()
    perplexity = Perplexity(
        torch_device=torch_device,
        iree_device=iree_device,
        iree_hip_target=iree_hip_target,
        iree_hal_target_device=iree_hal_target_device,
        kv_cache_type=kv_cache_type,
        tensor_parallelism_size=tensor_parallelism_size,
        attention_kernel=attention_kernel,
        block_seq_stride=block_seq_stride,
    )

    perplexity.get_prompts(num_prompts=num_prompts)

    vmfb_path = perplexity.compile_model(
        weight_path_str
    )  # == move this and export to a single ppl's conftest function, return vmfb path
    # perplexity.load_model(weight_path, tokenizer, vmfb_path) # Remove
    ppl = perplexity.get_perplexity()

    end = time.time()
    total_time = round(end - start, 2)
    if total_time < 60:
        total_time = str(total_time) + " secs"
    else:
        total_time = str(round(total_time / 60, 2)) + " mins"
    logger.info(f" Total time taken: {total_time}")

    return ppl


def main(argv):
    parser = cli.create_parser()
    parser.add_argument(
        "--attention-kernel",
        type=str,
        default="decomposed",
        choices=["decomposed", "torch_sdpa"],
    )
    parser.add_argument(
        "--block-seq-stride",
        help="Block sequence stride for paged KV cache, must divide evenly into the context length",
        type=int,
        default=32,
    )
    parser.add_argument("--iree-device", help="List an IREE device (e.g., 'hip://0')")
    parser.add_argument(
        "--iree-hip-target",
        action="store",
        default="gfx942",
        help="Specify the iree-hip target version (e.g., gfx942)",
    )
    parser.add_argument(
        "--iree-hal-target-device",
        action="store",
        default="hip",
        help="Specify the iree-hal target device (e.g., hip, cpu)",
    )
    parser.add_argument("--kv-cache-type", default="paged", help="KV cache type")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts for perplexity test (1 to 100)",
    )
    parser.add_argument(
        "--tensor-parallelism-size",
        type=int,
        default=1,
        help="Number of devices for tensor parallel sharding",
    )
    parser.add_argument("--torch-device", help="Torch device (or default)")

    cli.add_tokenizer_options(parser)
    cli.add_input_dataset_options(parser)
    args = cli.parse(parser, args=argv)

    torch_device = torch.device(args.torch_device) if args.torch_device else None
    weight_path = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)

    ppl = run_perplexity(
        weight_path=weight_path,
        weight_path_str=str(args.irpa_file),
        tokenizer=tokenizer,
        torch_device=torch_device,
        iree_device=args.iree_device,
        iree_hip_target=args.iree_hip_target,
        iree_hal_target_device=args.iree_hal_target_device,
        kv_cache_type=args.kv_cache_type,
        tensor_parallelism_size=args.tensor_parallelism_size,
        attention_kernel=args.attention_kernel,
        num_prompts=args.num_prompts,
        block_seq_stride=args.block_seq_stride,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
