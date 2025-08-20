# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import torch

from typing import Optional, Tuple

from sharktank import ops
from sharktank.layers import LlamaModelConfig
from sharktank.models.llm import PagedLlmModelV1
from sharktank.models.llm.config import ExportConfig, KVCacheConfig, ServiceConfig


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

    def allocate_cache(self, page_count: int):
        return self.model.cache.allocate(page_count=page_count)

    def prefill(self, tokens, start_pos, seq_lens, seq_block_ids, cs):
        cache_tensors = cs

        attention_mask = None
        if self.config.use_attention_mask:
            sl = tokens.shape[1]
            input_mask = self.model.input_mask(seq_lens, sl)
            attention_mask = self.model.attention_mask(
                input_mask, start_positions=start_pos
            )

        logits = self.model.prefill(
            tokens,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids,
            cache_state=cache_tensors,
            start_positions=start_pos,
        )

        logits = ops.unshard(logits)

        if self.config.logits_normalization == "softmax":
            logits = ops.softmax(logits, dim=-1)

        if self.config.logits_normalization == "log_softmax":
            logits = ops.elementwise(torch.log, ops.softmax(logits, dim=-1))

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
        cache_state,
    ):
        input_mask = self.model.input_mask(
            seq_lens, seq_block_ids.shape[1] * self.model.cache.block_seq_stride
        )
        attention_mask = self.model.decode_attention_mask(input_mask)

        logits = self.model.decode(
            tokens,
            attention_mask=attention_mask,
            start_positions=start_positions,
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


def build_service_config(
    llama_config: LlamaModelConfig, export_config: ExportConfig
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

    kv_config = KVCacheConfig(
        attention_head_count_kv=hp.attention_head_count_kv,
        block_seq_stride=llama_config.block_seq_stride,
        device_block_count=export_config.device_block_count,
        kv_cache_dtype=kv_cache_dtype,
    )

    return ServiceConfig(
        module_name="module",
        module_abi_version=1,
        max_seq_len=hp.context_length,
        attn_head_dim=hp.attn_head_dim,
        prefill_batch_sizes=export_config.bs_prefill,
        decode_batch_sizes=export_config.bs_decode,
        transformer_block_count=hp.block_count,
        logits_normalization=export_config.logits_normalization,
        top_k=export_config.top_k,
        paged_kv_cache=kv_config,
    )
