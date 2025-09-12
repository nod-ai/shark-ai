# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class KVCacheConfig:
    attention_head_count_kv: int
    block_seq_stride: int
    device_block_count: int
    kv_cache_dtype: int
    paged_kv_block_size_elements_per_device: list[int] | None = None


@dataclass
class ServiceConfig:
    module_name: str
    module_abi_version: int
    max_seq_len: int
    attn_head_dim: int
    prefill_batch_sizes: list[int]
    has_prefill_position: bool
    decode_batch_sizes: list[int]
    transformer_block_count: int
    logits_normalization: Optional[str]
    top_k: Optional[int]
    paged_kv_cache: KVCacheConfig

    @staticmethod
    def load(fp: Path):
        with open(fp, "rt") as f:
            server_config = ServiceConfig(**json.loads(f.read()))
            server_config.paged_kv_cache = KVCacheConfig(**server_config.paged_kv_cache)
        return server_config


@dataclass
class ExportConfig:
    device_block_count: int = 512
    top_k: Optional[int] = None
    logits_normalization: Optional[str] = None
    prefill_final_logits: bool = False
    use_linalgext_topk: bool = True
    has_prefill_position: Optional[bool] = False
    use_extend_attention: Optional[bool] = False

    bs_prefill: list[int] = field(default_factory=lambda: [4])
    bs_decode: list[int] = field(default_factory=lambda: [32])
    skip_prefill: bool = False
    skip_decode: bool = False
