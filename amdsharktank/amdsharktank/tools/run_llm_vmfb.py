# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import tokenizers

from amdsharktank.models.llm.config import ServiceConfig, KVCacheConfig
from amdsharktank.utils.llm_utils import (
    IreeInstance,
    LlmInstance,
    server_config_page_size,
)


class Tokenizer:
    def __init__(self, tokenizer_fp, config_fp):
        with open(config_fp, "rt") as f:
            config = json.loads(f.read())
            eos_token = config["eos_token"]
        self.t = tokenizers.Tokenizer.from_file(tokenizer_fp)
        self._eos_token_id = self.t.token_to_id(eos_token)

    @property
    def eos(self):
        return self._eos_token_id

    def encode(self, texts: list[str]) -> list[list[int]]:
        """Encodes a batch of texts, applying no padding."""
        return [s.ids for s in self.t.encode_batch(texts)]

    def decode(self, sequences) -> list[str]:
        """Decodes a batch of sequences to text."""
        return self.t.decode_batch(sequences)


class Decoder:
    def __init__(self, *, vmfb_fp, config_fp, irpa_fp, chunk_block_size):
        with open(vmfb_fp, "rb") as f:
            vmfb_bytes = f.read()

        with open(config_fp, "rt") as f:
            self._server_config = ServiceConfig(**json.loads(f.read()))
            self._server_config.paged_kv_cache = KVCacheConfig(
                **self._server_config.paged_kv_cache
            )

        if chunk_block_size is not None:
            assert (
                self._server_config.has_prefill_position
            ), "Chunking requires exporting with `--has-prefill-position`"

        # Extract the running configuration:
        page_kv_cache = self._server_config.paged_kv_cache
        self._block_seq_stride = page_kv_cache.block_seq_stride
        self._block_count = page_kv_cache.device_block_count
        self._page_sizes = server_config_page_size(self._server_config)

        devices = [f"hip://{i}" for i in range(len(self._page_sizes))]
        self._iree = IreeInstance(devices=devices, vmfb=vmfb_bytes, parameters=irpa_fp)
        self._llm = LlmInstance(
            self._iree,
            block_count=self._block_count,
            block_seq_stride=self._block_seq_stride,
            page_sizes=self._page_sizes,
            kv_cache_dtype=self._server_config.paged_kv_cache.kv_cache_dtype,
            chunk_block_size=chunk_block_size,
        )
        self._decoder = self._llm.make_decoder()

    def decode(self, *, tokens: list[list[int]], steps: int, eos: int):
        tokens = self._decoder.greedy_decode(tokens, steps=steps, eos=eos)
        return tokens


def main(
    prompts, steps, vmfb, config, irpa, tokenizer, tokenizer_config, chunk_block_size
):
    tokenizer = Tokenizer(tokenizer, tokenizer_config)
    tokens = tokenizer.encode(prompts)
    decoder = Decoder(
        vmfb_fp=vmfb, config_fp=config, irpa_fp=irpa, chunk_block_size=chunk_block_size
    )
    selected = decoder.decode(tokens=tokens, steps=steps, eos=tokenizer.eos)
    responses = tokenizer.decode(selected)
    for i in range(len(selected)):
        prompt = prompts[i]
        response = responses[i]
        print(f"-------- Prompt {i + 1} ----------")
        print(prompt)
        print(f"-------- Response {i + 1} --------")
        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", help="String to decode", required=True, action="append"
    )
    parser.add_argument("--irpa", help="IRPA parameters file", required=True)
    parser.add_argument("--vmfb", help="vmfb file path", required=True)
    parser.add_argument("--config", help="json config file for server", required=True)
    parser.add_argument("--tokenizer", help="tokenizer json file", required=True)
    parser.add_argument(
        "--tokenizer_config", help="tokenizer config json file", required=True
    )
    parser.add_argument(
        "--steps", help="steps to perform decode", type=int, required=True
    )
    parser.add_argument(
        "--chunk_block_size", help="block size for chunking", type=int, default=None
    )
    args = parser.parse_args()
    main(
        prompts=args.prompt,
        steps=args.steps,
        irpa=args.irpa,
        vmfb=args.vmfb,
        config=args.config,
        tokenizer=args.tokenizer,
        tokenizer_config=args.tokenizer_config,
        chunk_block_size=args.chunk_block_size,
    )
