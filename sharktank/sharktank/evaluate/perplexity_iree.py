# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import logging
import json
import time
import random
from datetime import timedelta
from tqdm import tqdm

import numpy as np

from datasets import load_dataset

import torch
from torch.nn import CrossEntropyLoss

from sharktank.models.llama.llama import *
from sharktank.models.mixtral.mixtral import *
from sharktank.models.grok.grok import *

from ..models.llama.sharding import shard_theta

from sharktank.layers import *
from sharktank.types import *

from sharktank.utils import cli
from sharktank.utils.vmfb_runner import *
from sharktank.utils.load_llm import *
from sharktank.utils.create_cache import *
from sharktank.utils.export_artifacts import *

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
        iree_hal_target_backends,
        kv_cache_type,
        tensor_parallelism_size,
        attention_kernel,
    ):
        self.torch_device = torch_device
        self.iree_device = iree_device
        self.iree_hip_target = iree_hip_target
        self.iree_hal_target_backends = iree_hal_target_backends
        self.kv_cache_type = kv_cache_type
        self.activation_dtype = torch.float16
        self.attention_dtype = torch.float16
        self.tensor_parallelism_size = tensor_parallelism_size
        self.attention_kernel = attention_kernel

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            seconds = end - start
            time_taken = abs(timedelta(seconds=round(seconds)))

            if seconds < 1:
                time_taken = f" {seconds * 1000} ms"

            func_name = func.__name__
            if func_name == "get_perplexity":
                func_name = f"Total time to calculate perplexity"
            elif func_name == "compile_model":
                func_name = f"Total time to export and compile"
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
        self.weight_path_str = weight_path_str

        logger.info(f"Compiling: {self.weight_path_str}")

        export_artifacts = ExportArtifacts(
            irpa_path=self.weight_path_str,
            batch_size=self.bs,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_backends=self.iree_hal_target_backends,
            attention_kernel=self.attention_kernel,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        vmfb_path = export_artifacts.get_artifacts()
        return vmfb_path

    @timeit
    def load_model(self, weight_path, tokenizer, vmfb_path):

        config = LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(weight_path.properties),
            block_seq_stride=16,
            kv_cache_type=self.kv_cache_type,
            device=self.torch_device,
            activation_dtype=self.activation_dtype,
            attention_dtype=self.attention_dtype,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )

        if config.tensor_parallelism_size > 1:
            weight_path.root_theta = shard_theta(weight_path.root_theta, config)

        theta = weight_path.root_theta

        if config.hp.expert_count:
            if config.hp.model_arch == "grok":
                model = PagedGrokModelV1(theta, config)
            else:
                model = PagedMixtralModelV1(theta, config)
        else:
            model = PagedLlamaModelV1(theta, config)

        self.generator = TorchGenerator(model, tokenizer)

        self.runner = vmfbRunner(
            device=self.iree_device,
            vmfb_path=vmfb_path,
            external_weight_path=self.weight_path_str,
        )

        self.haldevice = self.runner.config.device

    @timeit
    def get_prompts(self):
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
        ]

        self.bs = len(test_prompts)

        return test_prompts

    def prefill_vmfb(self, token_batch, i):

        seq_block_ids = self.batch.pad_block_ids()
        prefill_logits = self.runner.ctx.modules.module[f"prefill_bs{self.bs}"](
            token_batch,
            self.batch.seq_lens,
            seq_block_ids,
            self.cache_state,
        )

        prefill_logits = torch.tensor(prefill_logits[:, :, :])

        tokens = torch.tensor(
            self.generator.model.extract_tokens_from_logits(
                prefill_logits, self.batch.seq_lens
            )
        ).unsqueeze(1)
        self.batch.add_result_token(tokens)

        self.print_token_comparison(i)
        return prefill_logits

    def decode_vmfb(self, token_batch, i):
        logger.debug("Decode:")

        logger.debug("Input:")
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

        decode_logits = torch.tensor(decode_logits[:, :, :])

        tokens = torch.tensor(
            self.generator.model.extract_tokens_from_logits(
                decode_logits, [1] * self.bs
            ),
            device=self.generator.model.device,
        ).unsqueeze(1)
        self.batch.add_result_token(tokens)
        self.print_token_comparison(i)
        return decode_logits

    @timeit
    def get_logits(self):

        token_ids, seq_lens = self.generator.tokenizer.encode(
            self.test_prompts,
            pad_to_multiple_of=self.generator.model.cache.pad_sequence_stride,
        )

        logger.info(f" Prompts for Evaluation:")
        for idx, prompt in enumerate(self.test_prompts):
            logger.info(
                f" Prompt {idx}: \nTokens: {prompt.encode()}\nToken ids: {token_ids[idx]}\n"
            )

        self.max_prompt_length = max(seq_lens)

        self.token_ids = torch.tensor(token_ids, device=self.torch_device)
        self.attention_mask = (
            (self.token_ids != 0).int().detach().clone().to(self.torch_device)
        )

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
    def get_perplexity(self, test_prompts):

        self.test_prompts = test_prompts

        self.get_logits()

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
    iree_hal_target_backends,
    kv_cache_type,
    tensor_parallelism_size,
    attention_kernel,
):
    perplexity = Perplexity(
        torch_device=torch_device,
        iree_device=iree_device,
        iree_hip_target=iree_hip_target,
        iree_hal_target_backends=iree_hal_target_backends,
        kv_cache_type=kv_cache_type,
        tensor_parallelism_size=tensor_parallelism_size,
        attention_kernel=attention_kernel,
    )

    test_prompts = perplexity.get_prompts()
    logger.info(f" Total test prompts: {len(test_prompts)}")

    vmfb_path = perplexity.compile_model(weight_path_str)
    perplexity.load_model(weight_path, tokenizer, vmfb_path)
    ppl = perplexity.get_perplexity(test_prompts)

    return ppl


def main(argv):
    parser = cli.create_parser()
    parser.add_argument("--kv-cache-type", default="paged", help="KV cache type")
    parser.add_argument("--torch-device", help="Torch device (or default)")
    parser.add_argument("--iree-device", help="List an IREE device, eg: 'hip://0'")
    parser.add_argument(
        "--iree-hip-target",
        action="store",
        default="gfx942",
        help="Specify the iree-hip target version (e.g., gfx942)",
    )
    parser.add_argument(
        "--iree-hal-target-backends",
        action="store",
        default="rocm",
        help="Specify the iree-hal target backends (e.g., rocm)",
    )
    parser.add_argument(
        "--attention-kernel",
        type=str,
        default="decomposed",
        choices=["decomposed", "torch_sdpa"],
    )
    parser.add_argument(
        "--tensor-parallelism-size",
        type=int,
        default=1,
        help="Number of devices for tensor parallel sharding",
    )

    cli.add_tokenizer_options(parser)
    cli.add_input_dataset_options(parser)
    args = cli.parse(parser, args=argv)

    torch_device = torch.device(args.torch_device) if args.torch_device else None
    iree_device = args.iree_device
    kv_cache_type = args.kv_cache_type
    weight_path = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)
    weight_path_str = str(args.irpa_file)

    ppl = run_perplexity(
        weight_path=weight_path,
        weight_path_str=weight_path_str,
        tokenizer=tokenizer,
        torch_device=torch_device,
        iree_device=iree_device,
        iree_hip_target=args.iree_hip_target,
        iree_hal_target_backends=args.iree_hal_target_backends,
        kv_cache_type=kv_cache_type,
        tensor_parallelism_size=args.tensor_parallelism_size,
        attention_kernel=args.attention_kernel,
    )

    logger.info(f"\n{json.dumps(ppl, indent=2)}")
    return ppl


if __name__ == "__main__":
    main(sys.argv[1:])
