# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import dataclasses
import json
import logging
import math

from amdsharktank.models.llm.config import ServiceConfig, KVCacheConfig
from amdsharktank.utils.llm_utils import (
    IreeInstance,
    LlmInstance,
    server_config_page_size,
)


class Bencher:
    def __init__(self, *, vmfb_fp, config_fp, irpa_fp, total_length):

        with open(vmfb_fp, "rb") as f:
            vmfb_bytes = f.read()

        with open(config_fp, "rt") as f:
            self._server_config = ServiceConfig(**json.loads(f.read()))
            self._server_config.paged_kv_cache = KVCacheConfig(
                **self._server_config.paged_kv_cache
            )

        # Extract the running configuration:
        page_kv_cache = self._server_config.paged_kv_cache
        self._block_seq_stride = page_kv_cache.block_seq_stride
        self._block_count = page_kv_cache.device_block_count
        self._page_sizes = server_config_page_size(self._server_config)

        required_blocks = math.ceil(total_length / self._block_seq_stride)
        required_blocks = required_blocks * self._server_config.decode_batch_sizes[-1]
        if required_blocks >= self._block_count:
            logging.log(
                logging.ERROR,
                f"Required blocks ({required_blocks + 1}) exceeds exported ({self._block_count}) size. Increasing to required count.",
            )
            self._block_count = required_blocks + 1

        devices = [f"hip://{i}" for i in range(len(self._page_sizes))]
        self._iree = IreeInstance(devices=devices, vmfb=vmfb_bytes, parameters=irpa_fp)
        self._llm = LlmInstance(
            self._iree,
            block_count=self._block_count,
            block_seq_stride=self._block_seq_stride,
            page_size=self._page_size,
            kv_cache_dtype=page_kv_cache.kv_cache_dtype,
        )
        self._bencher = self._llm.make_bencher()

    def bench(self, *, length: int, steps: int):
        results = self._bencher.greedy_bench(length=length, steps=steps)
        return results


def main(length, steps, vmfb, config, irpa):
    total_length = length + steps
    decoder = Bencher(
        vmfb_fp=vmfb, config_fp=config, irpa_fp=irpa, total_length=total_length
    )
    results = decoder.bench(length=length, steps=steps)
    print(json.dumps(dataclasses.asdict(results), indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", help="Context Length", type=int, required=True)
    parser.add_argument("--irpa", help="IRPA parameters file", required=True)
    parser.add_argument("--vmfb", help="vmfb file path", required=True)
    parser.add_argument("--config", help="json config file for server", required=True)
    parser.add_argument(
        "--steps", help="steps to perform decode", type=int, required=True
    )
    args = parser.parse_args()
    main(
        length=args.length,
        steps=args.steps,
        irpa=args.irpa,
        vmfb=args.vmfb,
        config=args.config,
    )
