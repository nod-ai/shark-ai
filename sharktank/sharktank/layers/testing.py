# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union

import torch
from ..types.theta import Theta
from ..types.tensors import DefaultPrimitiveTensor, SplitPrimitiveTensor
from ..utils.testing import make_rand_torch
from ..utils.create_cache import create_kv_cache
from ..layers import BaseCausalLMModel, LlamaModelConfig
from ..utils.iree import (
    run_iree_module_function,
    prepare_iree_module_function_args,
    iree_to_torch,
)
import iree.runtime


def make_llama_attention_block_theta(
    *,
    block_idx: int,
    head_count: int,
    head_count_kv: int,
    head_dim: int,
    embedding_length: int,
    dtype: torch.dtype | None = None,
) -> Theta:
    return Theta(
        {
            "attn_q.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_q.weight",
                data=make_rand_torch(
                    (head_count * head_dim, embedding_length), dtype=dtype
                ),
            ),
            "attn_k.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_k.weight",
                data=make_rand_torch(
                    (head_count_kv * head_dim, embedding_length), dtype=dtype
                ),
            ),
            "attn_v.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_v.weight",
                data=make_rand_torch(
                    (head_count_kv * head_dim, embedding_length), dtype=dtype
                ),
            ),
            "attn_output.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_output.weight",
                data=make_rand_torch((embedding_length, embedding_length), dtype=dtype),
            ),
            "attn_norm.weight": DefaultPrimitiveTensor(
                name=f"blk.{block_idx}.attn_norm.weight",
                data=make_rand_torch((embedding_length), dtype=dtype),
            ),
        }
    )


def make_mmdit_double_block_random_theta(
    in_channels: int = 128,
    hidden_size: int = 3072,
    mlp_ratio: float = 4.0,
    dtype: torch.dtype | None = None,
) -> Theta:
    in_channels = 128
    hidden_size = 3072
    mlp_ratio = 4.0
    mlp_hidden_size = int((mlp_ratio - 1) * hidden_size)
    mlp_hidden_size2 = int(mlp_ratio * hidden_size)
    mlp_hidden_size3 = int(2 * (mlp_ratio - 1) * hidden_size)
    return Theta(
        {
            "img_attn.norm.key_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((in_channels,), dtype=dtype)
            ),
            "img_attn.norm.query_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((in_channels,), dtype=dtype)
            ),
            "img_attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size,), dtype=dtype)
            ),
            "img_attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
            ),
            "img_attn.qkv.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size,), dtype=dtype)
            ),
            "img_attn.qkv.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
            ),
            "img_mlp.0.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size2), dtype=dtype)
            ),
            "img_mlp.0.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size2, hidden_size), dtype=dtype)
            ),
            "img_mlp.2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size), dtype=dtype)
            ),
            "img_mlp.2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, mlp_hidden_size2), dtype=dtype)
            ),
            "img_mod.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size3,), dtype=dtype)
            ),
            "img_mod.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size3, hidden_size), dtype=dtype)
            ),
            "txt_attn.norm.key_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((in_channels,), dtype=dtype)
            ),
            "txt_attn.norm.query_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((in_channels,), dtype=dtype)
            ),
            "txt_attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size,), dtype=dtype)
            ),
            "txt_attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
            ),
            "txt_attn.qkv.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size,), dtype=dtype)
            ),
            "txt_attn.qkv.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
            ),
            "txt_mlp.0.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size2), dtype=dtype)
            ),
            "txt_mlp.0.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size2, hidden_size), dtype=dtype)
            ),
            "txt_mlp.2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size), dtype=dtype)
            ),
            "txt_mlp.2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, mlp_hidden_size2), dtype=dtype)
            ),
            "txt_mod.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size3,), dtype=dtype)
            ),
            "txt_mod.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size3, hidden_size), dtype=dtype)
            ),
        }
    )


def make_mmdit_single_block_random_theta(
    in_channels: int = 128,
    hidden_size: int = 3072,
    mlp_ratio: float = 4.0,
    dtype: torch.dtype | None = None,
) -> Theta:
    in_channels = 128
    hidden_size = 3072
    mlp_ratio = 4.0
    mlp_hidden_size = int((mlp_ratio - 1) * hidden_size)
    mlp_hidden_size2 = int((mlp_ratio + 1) * hidden_size)
    mlp_hidden_size3 = int((2 * mlp_ratio - 1) * hidden_size)
    return Theta(
        {
            "norm.key_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((in_channels,), dtype=dtype)
            ),
            "norm.query_norm.scale": DefaultPrimitiveTensor(  #
                data=make_rand_torch((in_channels,), dtype=dtype)
            ),
            "attn.proj.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size,), dtype=dtype)
            ),
            "attn.proj.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, hidden_size), dtype=dtype)
            ),
            "linear1.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size3,), dtype=dtype)
            ),
            "linear1.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size3, hidden_size), dtype=dtype)
            ),
            "linear2.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size), dtype=dtype)
            ),
            "linear2.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((hidden_size, mlp_hidden_size2), dtype=dtype)
            ),
            "modulation.lin.bias": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size,), dtype=dtype)
            ),
            "modulation.lin.weight": DefaultPrimitiveTensor(
                data=make_rand_torch((mlp_hidden_size, hidden_size), dtype=dtype)
            ),
        }
    )


class CausalLMIreeModel(BaseCausalLMModel):
    """Implements BaseCausalLMModel where prefill_from_seq_lens and decode_from_seq_lens
    are backed by an IREE module.
    This is meant only for testing. It may not be performant as it marshals tensors and
    uses the IREE Python bindings."""

    def __init__(
        self,
        config: LlamaModelConfig,
        batch_size: int,
        vm_context: iree.runtime.VmContext,
        iree_driver: str,
        iree_module: iree.runtime.VmModule,
        iree_devices: list[iree.runtime.HalDevice],
    ):
        super().__init__(
            context_length=config.hp.context_length,
            device=config.device,
            activation_dtype=config.activation_dtype,
            attention_dtype=config.attention_dtype,
        )
        self.config = config
        self.cache = create_kv_cache(self.config)
        self.vm_context = vm_context
        self.iree_driver = iree_driver
        self.iree_module = iree_module
        self.iree_devices = iree_devices
        self.batch_size = batch_size

    def prefill_from_seq_lens(
        self,
        tokens: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        seq_block_ids: torch.Tensor,
        cache_state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
    ):
        iree_args = prepare_iree_module_function_args(
            args=[tokens, seq_lens, seq_block_ids, cache_state],
            devices=self.iree_devices,
        )
        iree_results = run_iree_module_function(
            args=iree_args,
            function_name=f"prefill_bs{self.batch_size}",
            module=self.iree_module,
            vm_context=self.vm_context,
            driver=self.iree_driver,
        )

        iree_cache_state_shards = iree_args[-self.config.tensor_parallelism_size :]
        iree_cache_state = SplitPrimitiveTensor(
            ts=iree_to_torch(*iree_cache_state_shards),
            shard_dim=1,
        )
        cache_state[0][...] = iree_cache_state

        iree_results = iree_to_torch(*iree_results)
        return iree_results[0]

    def decode_from_seq_lens(
        self,
        tokens: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        start_positions: torch.Tensor,
        seq_block_ids: torch.Tensor,
        cache_state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
    ):
        iree_args = prepare_iree_module_function_args(
            args=[tokens, seq_lens, start_positions, seq_block_ids, cache_state],
            devices=self.iree_devices,
        )
        iree_results = run_iree_module_function(
            args=iree_args,
            function_name=f"decode_bs{self.batch_size}",
            module=self.iree_module,
            vm_context=self.vm_context,
            driver=self.iree_driver,
        )

        iree_cache_state_shards = iree_args[-self.config.tensor_parallelism_size :]
        iree_cache_state = SplitPrimitiveTensor(
            ts=iree_to_torch(*iree_cache_state_shards),
            shard_dim=1,
        )
        cache_state[0][...] = iree_cache_state

        iree_results = iree_to_torch(*iree_results)
        return iree_results[0]
