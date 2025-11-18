# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import torch

from typing import Optional, Tuple

from iree.turbine.aot import DeviceAffinity

from amdsharktank import ops
from amdsharktank.layers import LlamaModelConfig, CacheAllocation, KVCache, PagedKVCache
from amdsharktank.models.llm import PagedLlmModelV1
from amdsharktank.models.llm.config import ExportConfig, KVCacheConfig, ServiceConfig
from amdsharktank.utils.attention import *


def argmax_output(
    logits: torch.Tensor, chunk_size: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = ops.argmax(logits, -1, chunk_size=chunk_size)
    indices_expanded = indices.unsqueeze(-1)

    max_logits = ops.gather(logits, dim=-1, index=indices_expanded)

    return max_logits, indices_expanded


def topk_output(
    logits: torch.Tensor, k: int, chunk_size: int, use_linalgext_topk: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.topk(
        logits,
        k=k,
        dim=-1,
        largest=True,
        sorted=not use_linalgext_topk,
        chunk_size=chunk_size,
        use_linalgext_topk=use_linalgext_topk,
    )


class ServicePagedLlmModelV1(torch.nn.Module):
    def __init__(self, model: PagedLlmModelV1, config: ExportConfig):
        super().__init__()
        self.model = model
        self.config = config

    @property
    def is_paged(self):
        return self.model.config.kv_cache_type == "paged"

    def prefill(
        self, tokens, start_pos, seq_lens, seq_block_ids, cache_state: CacheAllocation
    ):
        logits = self.model.prefill(
            tokens,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
            start_positions=start_pos,
        )

        logits = ops.unshard(logits)

        if self.config.logits_normalization == "softmax":
            logits = logits.to(dtype=torch.float32)
            logits = ops.softmax(logits, dim=-1)
            logits = logits.to(dtype=torch.float16)

        if self.config.logits_normalization == "log_softmax":
            logits = logits.to(dtype=torch.float32)
            logits = ops.elementwise(torch.log, ops.softmax(logits, dim=-1))
            logits = logits.to(dtype=torch.float16)

        if self.config.prefill_final_logits:
            last_seq_lens = seq_lens
            bsi = torch.tensor(list(range(logits.shape[0])))

            logits = logits[bsi, last_seq_lens - 1]
            logits = logits.unsqueeze(1)

        if self.config.top_k is None:
            return logits

        if self.config.top_k == 1:
            return argmax_output(logits, chunk_size=None)

        return topk_output(
            logits,
            k=self.config.top_k,
            chunk_size=256,
            use_linalgext_topk=self.config.use_linalgext_topk,
        )

    def decode(
        self,
        tokens,
        seq_lens,
        start_positions,
        seq_block_ids,
        cache_state: CacheAllocation,
    ):
        logits = self.model.decode(
            tokens,
            start_positions=start_positions,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            cache_state=cache_state,
        )

        logits = ops.unshard(logits)

        if self.config.logits_normalization == "softmax":
            logits = ops.softmax(logits, dim=-1)

        if self.config.logits_normalization == "log_softmax":
            logits = ops.elementwise(torch.log, ops.softmax(logits, dim=-1))

        if self.config.top_k is None:
            return logits

        if self.config.top_k == 1:
            return argmax_output(logits, chunk_size=None)

        return topk_output(
            logits,
            k=self.config.top_k,
            chunk_size=256,
            use_linalgext_topk=self.config.use_linalgext_topk,
        )

    def setup_arg_devices(
        self,
        cache_affinities: list[DeviceAffinity],
        num_input_args: int,
    ) -> dict[int, DeviceAffinity]:
        num_non_cache_args = num_input_args - 1  # Exclude cache state
        affinity_0 = self.model.config.parallelism_config.device_affinity_for_pipeline(
            0
        )

        arg_devices = [affinity_0 for _ in range(num_non_cache_args)]
        arg_devices.extend(cache_affinities)
        return {i: affinity for i, affinity in enumerate(arg_devices)}

    def setup_cache(
        self,
    ) -> tuple[
        list[torch.Tensor], list[dict[int, torch.export.Dim]], list[DeviceAffinity]
    ]:
        if not self.is_paged:
            raise NotImplementedError(f"Unsupported KV cache type")

        device_block_count = self.config.device_block_count
        cache_state = self.model.cache.allocate(page_count=device_block_count)
        page_dim = torch.export.Dim("page")

        unpacked = cache_state.allocation
        dynamic_shapes = [{0: page_dim} for _ in range(len(unpacked))]

        return unpacked, dynamic_shapes, cache_state.device_affinities


def build_service_config(
    llama_config: LlamaModelConfig, export_config: ExportConfig, kv_cache: KVCache
) -> ServiceConfig:
    """
    Generate config.json for shortfin.


    For shortfin, we only write attention_head_count_kv because that's all shortfin needs.
    Note that this is different from hp.attn_head_count when grouped attention shares kvcache between heads.
    """
    hp = llama_config.hp

    kv_cache_dtype = (
        llama_config.attention_dtype
        if llama_config.kv_cache_dtype is None
        else llama_config.kv_cache_dtype
    )

    kv_cache_dtype = str(kv_cache_dtype).split(".")[-1]

    assert isinstance(kv_cache, PagedKVCache)

    kv_config = KVCacheConfig(
        attention_head_count_kv=hp.attention_head_count_kv,
        block_seq_stride=llama_config.block_seq_stride,
        device_block_count=export_config.device_block_count,
        kv_cache_dtype=kv_cache_dtype,
        paged_kv_block_size_elements_per_device=kv_cache.block_size_elements_per_device,
    )

    return ServiceConfig(
        module_name="module",
        module_abi_version=1,
        max_seq_len=hp.context_length,
        attn_head_dim=hp.attn_head_dim,
        prefill_batch_sizes=export_config.bs_prefill,
        has_prefill_position=export_config.has_prefill_position,
        decode_batch_sizes=export_config.bs_decode,
        transformer_block_count=hp.block_count,
        logits_normalization=export_config.logits_normalization,
        top_k=export_config.top_k,
        paged_kv_cache=kv_config,
    )
