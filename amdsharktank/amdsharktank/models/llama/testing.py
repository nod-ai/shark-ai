# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import functools
import torch
import re
from amdsharktank.types.tensors import *
from amdsharktank.types import DynamicFp4BlockQuantizer, StaticScaledQuantizer
from amdsharktank.types.theta import Theta
from amdsharktank.layers.configs import LlamaModelConfig
from amdsharktank.utils.random import make_rand_torch
from amdsharktank.layers.testing import (
    make_llama_attention_block_theta,
    make_ffn_block_theta,
    make_random_moe_block_theta,
)


def make_attention_block_theta(
    feature_dim: int,
    ffn_dim: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_k.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_v.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, feature_dim), dtype=dtype)
            ),
            "attn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
            "ffn_gate.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "ffn_up.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((ffn_dim, feature_dim), dtype=dtype)
            ),
            "ffn_down.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim, ffn_dim), dtype=dtype)
            ),
            "ffn_norm.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((feature_dim), dtype=dtype)
            ),
        }
    )


def make_attention_block_ffn_theta_v2(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    feed_forward_length: int,
    dtype_rest: torch.dtype,
    dtype_norm: torch.dtype,
) -> Theta:
    attention_theta = make_llama_attention_block_theta(
        block_idx=block_idx,
        head_count=head_count,
        head_count_kv=head_count_kv,
        head_dim=head_dim,
        embedding_length=embedding_length,
        dtype=dtype_rest,
        dtype_norm=dtype_norm,
    )
    ffn_theta = make_ffn_block_theta(
        block_idx=block_idx,
        embedding_length=embedding_length,
        feed_forward_length=feed_forward_length,
        dtype_norm=dtype_norm,
        dtype=dtype_rest,
    )
    res_dict = attention_theta.tree
    res_dict.update(ffn_theta.tree)
    return Theta(res_dict)


def make_attention_moe_block_random_theta(
    block_idx: int,
    config: LlamaModelConfig,
    dtype_rest: torch.dtype,
    dtype_norm: torch.dtype,
) -> Theta:
    res_dict = {}
    attention_theta = make_llama_attention_block_theta(
        block_idx=block_idx,
        head_count=config.hp.attention_head_count,
        head_count_kv=config.hp.attention_head_count_kv,
        head_dim=config.hp.attn_head_dim,
        embedding_length=config.hp.embedding_length,
        dtype=dtype_rest,
        dtype_norm=dtype_norm,
    )
    res_dict.update(attention_theta.tree)
    moe_theta = make_random_moe_block_theta(
        block_idx=block_idx,
        in_dim=config.hp.embedding_length,
        expert_hidden_dim=config.hp.expert_feed_forward_length,
        num_experts=config.hp.expert_count,
        with_ffn_norm=True,
        num_shared_experts=config.hp.expert_shared_count,
        with_layer_output_norm=False,
        dtype_rest=dtype_rest,
        dtype_norm=dtype_norm,
    )
    res_dict.update(moe_theta.tree)
    return Theta(res_dict)


def make_random_llama_theta(
    config: LlamaModelConfig,
    vocab_size: Optional[int] = None,
    dtype_rest: torch.dtype = torch.float16,
    dtype_norm: torch.dtype = torch.float32,
) -> Theta:
    if vocab_size is None:
        vocab_size = config.hp.vocab_size

    res = {
        "token_embd.weight": DefaultPrimitiveTensor(
            name="token_embd.weight",
            data=make_rand_torch(
                (vocab_size, config.hp.embedding_length), dtype=dtype_rest
            ),
        )
    }
    for i in range(config.hp.block_count):
        is_moe_block = i in config.moe_layers
        if is_moe_block:
            # This is used in Llama 4.
            block = make_attention_moe_block_random_theta(
                config=config, block_idx=i, dtype_rest=dtype_rest, dtype_norm=dtype_norm
            ).tree
        else:
            block = make_attention_block_ffn_theta_v2(
                block_idx=i,
                head_count=config.hp.attention_head_count,
                head_count_kv=config.hp.attention_head_count_kv,
                head_dim=config.hp.attn_head_dim,
                embedding_length=config.hp.embedding_length,
                feed_forward_length=config.hp.feed_forward_length,
                dtype_rest=dtype_rest,
                dtype_norm=dtype_norm,
            ).tree
        res[f"blk.{i}"] = block

    res[f"output.weight"] = DefaultPrimitiveTensor(
        name="output.weight",
        data=make_rand_torch(
            (vocab_size, config.hp.embedding_length), dtype=dtype_rest
        ),
    )
    res[f"output_norm.weight"] = DefaultPrimitiveTensor(
        name="output_norm.weight",
        data=make_rand_torch((1, config.hp.embedding_length), dtype=dtype_norm),
    )

    return Theta(res)


def quantize_theta_to_fp4(theta: Theta, quantizer: DynamicFp4BlockQuantizer) -> Theta:
    """Quantize a Llama 3.1 model for with FP4 precision.

    This is not a serious quantization, just for testing with the toy model."""

    block_subnames_to_quantize = set(
        [
            "ffn_gate.weight",
            "ffn_down.weight",
            "ffn_up.weight",
            "attn_output.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
        ]
    )

    def quantize_tensor_to_fp4(
        tensor: InferenceTensor, quantizer: DynamicFp4BlockQuantizer
    ) -> QuantizedTensor:
        return quantizer.quantize(tensor, name=tensor.name)

    def should_quantize(fully_qualified_parameter_name: str) -> bool:
        match = re.match("^blk\\.[0-9]+\\.", fully_qualified_parameter_name)
        if not match:
            return False

        subname = re.sub("^blk\\.[0-9]+\\.", "", fully_qualified_parameter_name)
        return subname in block_subnames_to_quantize

    def quantize_transform(
        tensor: InferenceTensor,
    ) -> InferenceTensor | list[InferenceTensor]:
        if should_quantize(tensor.name):
            return quantize_tensor_to_fp4(tensor, quantizer=quantizer)
        assert isinstance(tensor, (PrimitiveTensor, StaticScaledQuantizer)), "TODO"
        return tensor

    def insert_kv_cache_quantizer_transform(
        tensor: InferenceTensor,
    ) -> InferenceTensor | list[InferenceTensor]:
        if tensor.name.endswith("attn_output.weight"):
            return tensor, StaticScaledQuantizer(
                name=tensor.name.replace("attn_output.weight", "kv_cache.quantizer"),
                scale=torch.tensor(0.5, dtype=tensor.dtype),
                dtype=torch.float8_e4m3fn,
            )
        return tensor

    return theta.transform(insert_kv_cache_quantizer_transform, quantize_transform)
