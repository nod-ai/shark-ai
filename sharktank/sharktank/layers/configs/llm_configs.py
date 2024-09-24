# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured configuration objects for various LLMs.

This draws heavily from the work that ggml has done to systematize the state
of the world for GGUF files:
  https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

When in question, we draw from the vocabulary and normalization they have done
(and indeed, can bootstrap these off of GGUF files).
"""

from dataclasses import dataclass
from typing import Any, Optional
from ...layers import *

import torch

__all__ = ["LlamaHParams", "LlamaModelConfig"]


@dataclass
class LlamaHParams:
    """Corresponds 1:1 with the 'LLM' section of the GGUF docs.

    Comments are only provided if they differ from this source.
    """

    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    rope_dimension_count: int
    rope_freq_base: float
    attention_head_count: int
    attn_head_dim: int
    attention_layer_norm_rms_epsilon: float
    attention_head_count_kv: int
    expert_count: int
    expert_used_count: int
    model_variant: str

    @staticmethod
    def from_gguf_props(p: dict[str, Any]):
        name_prefix = "llama"
        # These shouldnt be required for llama.
        default_attn_dim = None
        default_rope_count = None
        if "grok.attention.head_count" in p:
            name_prefix = "grok"
            default_attn_dim = 128
            default_rope_count = 128
        default_expert_count = 0
        default_expert_used_count = 0
        default_rope_freq_base = 10000.0
        attention_head_count = _int_prop(p, f"{name_prefix}.attention.head_count")

        return LlamaHParams(
            context_length=_int_prop(p, f"{name_prefix}.context_length"),
            embedding_length=_int_prop(p, f"{name_prefix}.embedding_length"),
            block_count=_int_prop(p, f"{name_prefix}.block_count"),
            feed_forward_length=_int_prop(p, f"{name_prefix}.feed_forward_length"),
            attn_head_dim=_optional_int_prop(
                p, f"{name_prefix}.rope.dimension_count", default_attn_dim
            ),
            rope_dimension_count=_optional_int_prop(
                p, f"{name_prefix}.rope.dimension_count", default_rope_count
            ),
            attention_head_count=attention_head_count,
            attention_layer_norm_rms_epsilon=_float_prop(
                p, f"{name_prefix}.attention.layer_norm_rms_epsilon"
            ),
            attention_head_count_kv=_optional_int_prop(
                p, f"{name_prefix}.attention.head_count_kv", attention_head_count
            ),
            rope_freq_base=_optional_float_prop(
                p, f"{name_prefix}.rope.freq_base", default_rope_freq_base
            ),
            expert_count=_optional_int_prop(
                p, f"{name_prefix}.expert_count", default_expert_count
            ),
            expert_used_count=_optional_int_prop(
                p, f"{name_prefix}.expert_used_count", default_expert_used_count
            ),
            model_variant=name_prefix,
        )


def _float_prop(p: dict[str, Any], name: str) -> float:
    try:
        return float(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _int_prop(p: dict[str, Any], name: str) -> int:
    try:
        return int(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _optional_float_prop(p: dict[str, Any], name: str, default_value: float) -> float:
    value = p.get(name, default_value)
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e


def _optional_int_prop(p: dict[str, Any], name: str, default_value: int) -> int:
    value = p.get(name, default_value)
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e


@dataclass
class LlamaModelConfig:
    hp: LlamaHParams

    # Block sequence stride for a paged KV cache. This must divide evenly
    # into the context length.
    block_seq_stride: int = 16

    # Either "paged" or "direct".
    kv_cache_type: str = "paged"

    # The device on which to place intermediate state.
    device: Optional[torch.device] = None

    # Dtype to use for general FP activations not otherwise configured.
    activation_dtype: torch.dtype = torch.float16

    # Dtype to use for attention.
    attention_dtype: torch.dtype = torch.float16

    # Indicates if running with HuggingFace implementation and ensures
    # numerical equivalency to HuggingFace's LLaMa if true (by modifying
    # rotary embedding).
    use_hf: bool = False

    # If true, then the model may pre-initialize certain tables during
    # init. This can be better for eager execution but when capturing a program,
    # it is often better to preserve the calculation explicitly and rely on
    # the compiler to transform it to an initialization time step. This can
    # be the difference of many gigabytes of static data being embedded in
    # the program and not.
    static_tables: bool = True

    def create_kv_cache(self) -> BaseKVCache:
        hp = self.hp
        if self.kv_cache_type == "direct":
            return DirectKVCache(
                block_seq_stride=self.block_seq_stride,
                transformer_block_count=hp.block_count,
                attn_head_count=hp.attention_head_count_kv,
                attn_head_dim=hp.attn_head_dim,
                seq_length=hp.context_length,
                device=self.device,
                dtype=self.attention_dtype,
            )
        elif self.kv_cache_type == "paged":
            return PagedKVCache(
                transformer_block_count=hp.block_count,
                attn_head_count=hp.attention_head_count_kv,
                attn_head_dim=hp.attn_head_dim,
                cache_partition_count=2,  # One for each of K/V.
                block_seq_stride=self.block_seq_stride,
                device=self.device,
                dtype=self.attention_dtype,
            )
        else:
            raise NotImplementedError(f"kv_cache_type = {self.kv_cache_type}")
