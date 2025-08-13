# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import tokenizers

from sharktank.utils.llm_utils import IreeInstance, LlmInstance, server_config_page_size


class Tokenizer:
    def __init__(self, fp):
        self.t = tokenizers.Tokenizer.from_file(fp)

    def encode(self, texts: list[str]) -> list[list[int]]:
        """Encodes a batch of texts, applying no padding."""
        return [s.ids for s in self.t.encode_batch(texts)]

    def decode(self, sequences) -> list[str]:
        """Decodes a batch of sequences to text."""
        return self.t.decode_batch(sequences)


class Decoder:
    def __init__(self, *, vmfb_fp, config_fp, irpa_fp):

        with open(vmfb_fp, "rb") as f:
            vmfb_bytes = f.read()

        with open(config_fp, "rt") as f:
            self._server_config = json.loads(f.read())

        # Extract the running configuration:
        page_kv_cache = self._server_config["paged_kv_cache"]
        self._block_seq_stride = page_kv_cache["block_seq_stride"]
        self._block_count = page_kv_cache["device_block_count"]
        self._page_size = server_config_page_size(self._server_config)

        self._iree = IreeInstance(
            devices=["hip://0"], vmfb=vmfb_bytes, parameters=irpa_fp
        )
        self._llm = LlmInstance(
            self._iree,
            block_count=self._block_count,
            block_seq_stride=self._block_seq_stride,
            page_size=self._page_size,
        )
        self._decoder = self._llm.make_decoder()

    def decode(self, *, tokens: list[int], steps: int):
        tokens = self._decoder.greedy_decode([tokens], steps=steps)
        return tokens


def main(prompt, steps, vmfb, config, irpa, tokenizer):
    tokenizer = Tokenizer(tokenizer)
    ids = tokenizer.encode([prompt])
    decoder = Decoder(vmfb_fp=vmfb, config_fp=config, irpa_fp=irpa)
    tokens = ids[0]

    selected = decoder.decode(tokens=tokens, steps=steps)
    print(tokenizer.decode(selected)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="String to decode", required=True)
    parser.add_argument("--irpa", help="IRPA parameters file", required=True)
    parser.add_argument("--vmfb", help="vmfb file path", required=True)
    parser.add_argument("--config", help="json config file for server", required=True)
    parser.add_argument("--tokenizer", help="json tokenizer config file", required=True)
    parser.add_argument(
        "--steps", help="steps to perform decode", type=int, required=True
    )
    args = parser.parse_args()
    main(
        prompt=args.prompt,
        steps=args.steps,
        irpa=args.irpa,
        vmfb=args.vmfb,
        config=args.config,
        tokenizer=args.tokenizer,
    )
