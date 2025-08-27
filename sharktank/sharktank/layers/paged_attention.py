# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Light-weight encapsulations for various forms of attention KV-caches.

These are not complete abstractions: they are primarily focused on making
tightly coupled transformer blocks a bit less "stringy" with loose tensors
and dims floating around everywhere.
"""

from typing import Optional, Union, List

import math

import torch
from collections import defaultdict
from sharktank.types import (
    DefaultPrimitiveTensor,
    QuantizerTensor,
    PlanarQuantizedTensor,
    ShardedTensor,
    StaticScaledQuantizer,
    TensorScaledLayout,
)
from sharktank import ops, kernels
from sharktank.kernels.mlir_kernel import *
from sharktank.types.tensors import AnyTensor

__all__ = ["PagedAttention", "attn_type_map", "CacheAllocation"]

attn_type_map = defaultdict(lambda: "gqa")
attn_type_map.update(
    {
        "llama": "gqa",
        "grok": "gqa",
        "deepseek2": "mla",
        "llama4": "gqa",
    }
)


# Paged Attention Kernels
#
# Each kernel is put into its own class to create a namespace for it
def KVCacheGatherKernel():
    CACHE_SIZE = DynDim.CACHE_SIZE
    PAGES = DynDim.PAGES
    T_BLOCK = StaticDim.T_BLOCK
    PART = StaticDim.PART
    BLOCK_SEQ_STRIDE = StaticDim.BLOCK_SEQ_STRIDE
    HEAD_COUNT_KV = StaticDim.HEAD_COUNT_KV
    ATTN_HEAD_DIM = StaticDim.ATTN_HEAD_DIM
    BATCH = DynDim.BATCH

    CACHE_TY = Dtype.CACHE_TY
    I64 = Dtype.I64

    @mlir_kernel(
        inputs=(
            MLIRTensor[
                CACHE_SIZE,
                T_BLOCK,
                PART,
                HEAD_COUNT_KV,
                BLOCK_SEQ_STRIDE,
                ATTN_HEAD_DIM,
                CACHE_TY,
            ],
            MLIRTensor[BATCH, PAGES, I64],
            MLIRTensor[I64],
            MLIRTensor[I64],
        ),
        results=(
            MLIRTensor[
                BATCH, PAGES, HEAD_COUNT_KV, BLOCK_SEQ_STRIDE, ATTN_HEAD_DIM, CACHE_TY
            ],
        ),
    )
    def paged_attention_kv_cache_gather(
        cache, page_ids, transformer_idx, partition_idx, result
    ):
        mlir = """
        !cache_slice = tensor<{{[CACHE_SIZE, HEAD_COUNT_KV, BLOCK_SEQ_STRIDE, ATTN_HEAD_DIM]|join('x')}}x!cache_dtype>

        module {
        util.func private @{{kernel_name}}(%cache: !cache,
                                   %page_ids: !page_ids,
                                   %transformer_idx: !transformer_idx,
                                   %partition_idx: !partition_idx) -> !result {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index

          // Get transformer/partition ids.
          %t_id64 = tensor.extract %transformer_idx[] : !transformer_idx
          %p_id64 = tensor.extract %partition_idx[] : !partition_idx
          %t_id = arith.index_cast %t_id64 : !transformer_idx_dtype to index
          %p_id = arith.index_cast %p_id64 : !partition_idx_dtype to index

          // Get dynamic dimensions.
          %cache_size = tensor.dim %cache, %c0 : !cache
          %batches = tensor.dim %page_ids, %c0 : !page_ids
          %pages = tensor.dim %page_ids, %c1 : !page_ids

          // Extract a the current transformer block and partition from cache.
          %cache_slice = tensor.extract_slice %cache
            [0, %t_id, %p_id, 0, 0, 0]
            [%cache_size, 1, 1, {{HEAD_COUNT_KV}}, {{BLOCK_SEQ_STRIDE}}, {{ATTN_HEAD_DIM}}]
            [1, 1, 1, 1, 1, 1]
            : !cache to !cache_slice

          %empty = tensor.empty(%batches, %pages) : !result

          // Gather from cache_slice using page_ids.
          %result = iree_linalg_ext.gather
                    dimension_map = [0]
                    ins(%cache_slice, %page_ids : !cache_slice, !page_ids)
                    outs(%empty : !result) -> !result

          util.return %result : !result
        }
        }
        """
        return MLIRSpec(mlir)

    return paged_attention_kv_cache_gather


kv_cache_gather = KVCacheGatherKernel()


def unpack_to_raw_tensor(tensor: AnyTensor) -> AnyTensor:
    """
    Unpacks the input tensor to a torch tensor if is a planar quantized tensor.
    If the input is a sharded tensor containing planar quantized tensors, it unpacks
    each shard and returns a new sharded tensor with the unpacked shards.
    """
    if isinstance(tensor, PlanarQuantizedTensor):
        return tensor.unpack()._qs

    return tensor


def pack_raw_tensor(tensor, quantizer):
    if quantizer is None:
        return tensor
    layout = TensorScaledLayout(
        shape=tensor.shape,
        d=quantizer._reciprocal_scale,
        qs=tensor,
        m=quantizer._offset,
    )
    return PlanarQuantizedTensor(shape=tensor.shape, layout=layout)


class CacheAllocation:
    def __init__(self, allocation: list[torch.Tensor]):
        self.allocation = allocation

    def __len__(self):
        return len(self.allocation)

    def __getitem__(self, idx):
        return self.allocation[idx]


class KVCache:
    def __init__(
        self,
        *,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        cache_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        devices: List[int] | None = None,
    ):
        self.transformer_block_count = transformer_block_count
        self.attn_head_count = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride
        self.cache_dtype = cache_dtype
        self.device = device
        self.devices = devices

        assert devices is None or len(devices) == 1
        assert cache_partition_count == 2

        # Some derived values based on attributes.
        self.sub_page_dims = [
            self.transformer_block_count,
            self.cache_partition_count,
            self.attn_head_count,
            self.block_seq_stride,
            self.attn_head_dim,
        ]

        self.page_slab_flat_dims = math.prod(self.sub_page_dims)

    def allocate(self, page_count: int) -> CacheAllocation:
        tensors = [
            torch.zeros(
                [page_count, self.page_slab_flat_dims],
                dtype=self.cache_dtype,
                device=self.device,
            )
        ]

        return CacheAllocation(tensors)

    @property
    def state_count(self):
        return 1

    def unflatten_page_table(self, state: CacheAllocation) -> List[torch.Tensor]:
        assert len(state) == 1
        """Unflattens the 2D page tables to 6D tensors."""
        return [state[0].unflatten(1, self.sub_page_dims)]

    def read(
        self,
        state: CacheAllocation,
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor,
    ):
        page_table = self.unflatten_page_table(state)[0]

        # TODO: mlir_kernel doesn't support non-tensor args yet, so use 0-D
        # tensors instead.
        t_id = torch.tensor(transformer_block_index, dtype=torch.int64)
        key_p_id = torch.tensor(0, dtype=torch.int64)
        value_p_id = torch.tensor(1, dtype=torch.int64)

        def unwrap_args(*ts):
            new_ts = []
            for t in ts:
                if isinstance(t, DefaultPrimitiveTensor):
                    t = t._data
                new_ts.append(t)
            return new_ts

        key = kv_cache_gather(*unwrap_args(page_table, page_ids, t_id, key_p_id))
        value = kv_cache_gather(*unwrap_args(page_table, page_ids, t_id, value_p_id))

        key = key.transpose(2, 3).flatten(1, 2)
        value = value.transpose(2, 3).flatten(1, 2)

        return key, value

    def write(
        self,
        *,
        state: CacheAllocation,
        cache_partitions: List[torch.Tensor],
        transformer_block_index: int,
        page_ids: torch.Tensor,
        start_positions: torch.Tensor,
    ):
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        assert len(state) == 1
        assert len(cache_partitions) == self.cache_partition_count

        page_table = self.unflatten_page_table(state=state)[0]
        page_table = page_table.flatten(0, 2)

        block_seq_len = cache_partitions[0].shape[1] // self.block_seq_stride

        if start_positions is not None:
            page_index = (
                start_positions.unsqueeze(1) // self.block_seq_stride
            ) + torch.arange(block_seq_len)
            page_ids = torch.gather(page_ids, dim=1, index=page_index)

        _, block_seq_len, *_ = page_ids.shape
        for cache_partition_id, cache_partition in enumerate(cache_partitions):
            index = page_ids
            index = index * self.transformer_block_count + transformer_block_index
            index = index * self.cache_partition_count + cache_partition_id
            index = index.flatten(0, 1)

            cache_partition = cache_partition.unflatten(
                1, (block_seq_len, self.block_seq_stride)
            )
            cache_partition = cache_partition.flatten(0, 1)
            cache_partition = cache_partition.transpose(1, 2)

            part_block = ops.to(cache_partition, dtype=page_table.dtype)
            ops.index_copy_(page_table, 0, index, part_block)

    def write_timestep(
        self,
        *,
        state: CacheAllocation,
        cache_partitions: List[torch.Tensor],
        transformer_block_index: int,
        seq_positions: torch.Tensor,
        page_ids: torch.Tensor,
    ):
        assert len(state) == 1
        assert len(cache_partitions) == self.cache_partition_count

        page_table = self.unflatten_page_table(state)[0]
        page_table = page_table.flatten(0, 4)

        device = self.device
        bs, *_ = seq_positions.shape

        page_index = seq_positions // self.block_seq_stride
        page_index = page_index.unsqueeze(1)
        page_id = ops.gather(page_ids, dim=1, index=page_index).view((bs, 1, 1))
        page_offset = (seq_positions % self.block_seq_stride).view((bs, 1, 1))
        head_offset = torch.arange(self.attn_head_count, device=device).view(
            (1, 1, self.attn_head_count)
        )

        for cache_partition_id, cache_partition in enumerate(cache_partitions):
            # [1, 1]
            partitions = torch.tensor(cache_partition_id, device=device).view((1, 1, 1))

            index = page_id
            index = index * self.transformer_block_count + transformer_block_index
            index = index * self.cache_partition_count + partitions
            index = index * self.attn_head_count + head_offset
            index = index * self.block_seq_stride + page_offset

            cache_partition.transpose(1, 2)
            values = ops.to(cache_partition, dtype=page_table.dtype)
            ops.index_put_(page_table, indices=(index,), values=values)


def build_cache(
    transformer_block_count: int,
    attn_head_count: int,
    attn_head_dim: int,
    devices: List[int] | None = None,
    cache_partition_count: int = 2,
    block_seq_stride: int = 16,
    cache_dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    return KVCache(
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        cache_partition_count=cache_partition_count,
        block_seq_stride=block_seq_stride,
        cache_dtype=cache_dtype,
        device=device,
        devices=devices,
    )


class PagedAttention:
    """Implementation of paged attention

    The page table slab is physically represented as a 2D tensor:
        [page_count, flattened_dims]

    Each "page" can be thought of as a 6D view onto:

    * transformer block
    * cache partition (K or V cache)
    * attention heads
    * block sequence stride (number of sequence positions per block)
    * attention dimensionality

    Note that the internal page structure matches the organization of the
    model, allowing contiguous individual local reads and writes at a sub-block
    granularity if indexing deeply into the structure.

    When `shard_count > 1`, it would split the `attn_head_count` dimension.
    The page slab is a 1D sharded split tensor.
    It is reinterpreted as a 6D tensor, by working around the lack of sharded
    block-cyclic sharded tensor type.
    """

    def __init__(
        self,
        *,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        attn_type: str = "gqa",
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        cache_dtype: torch.dtype = torch.float32,
        attn_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.transformer_block_count = transformer_block_count
        self.head_count_kv = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.block_seq_stride = block_seq_stride
        self.device = device
        self.attn_dtype = attn_dtype
        self.cache_dtype = cache_dtype
        self.attn_type = attn_type

        self.kv_cache = build_cache(
            transformer_block_count=transformer_block_count,
            attn_head_count=attn_head_count,
            attn_head_dim=attn_head_dim,
            cache_partition_count=cache_partition_count,
            block_seq_stride=block_seq_stride,
            cache_dtype=cache_dtype,
            device=device,
        )

    def shard_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.kv_cache.shard_state(state=state)

    def unshard_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.kv_cache.unshard_state(state=state)

    @property
    def pad_sequence_stride(self) -> int:
        return self.block_seq_stride

    def allocate(self, page_count: int) -> CacheAllocation:
        return self.kv_cache.allocate(page_count=page_count)

    def read(
        self,
        state: CacheAllocation,
        *,
        transformer_block_index: int,
        page_ids: Optional[torch.Tensor] = None,
    ):
        return self.kv_cache.read(
            state=state,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
        )

    def write_timestep(
        self,
        state: CacheAllocation,
        cache_partitions: List[torch.Tensor],
        transformer_block_index: int,
        seq_positions: torch.Tensor,
        page_ids: torch.Tensor,
    ):
        self.kv_cache.write_timestep(
            state=state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            seq_positions=seq_positions,
            page_ids=page_ids,
        )

    def write(
        self,
        state: CacheAllocation,
        cache_partitions: List[torch.Tensor],
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
    ):
        self.kv_cache.write(
            state=state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
            start_positions=start_positions,
        )

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        bs, slen, n_kv_heads, head_dim = x.shape
        unsq = x.unsqueeze(-2)
        exp = ops.expand(unsq, (bs, slen, n_kv_heads, n_rep, head_dim))
        return exp.flatten(2, 3)

    def gqa(self, head_count_attn, k, v):
        gqa_n_rep = head_count_attn // self.head_count_kv
        assert gqa_n_rep > 0
        if gqa_n_rep > 1:
            k = self.repeat_kv(x=k, n_rep=gqa_n_rep)
            v = self.repeat_kv(x=v, n_rep=gqa_n_rep)
        return k, v

    def attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        attention_kernel: str,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if self.attn_type == "gqa":
            k, v = self.gqa(head_count_attn, k, v)

        # Fake quant is already dequantized when stored in the cache.
        if cache_quantizer and not fake_quant:
            k_planes = {"qs": k}
            k = ops.dequantize(
                k_planes, quantizer=cache_quantizer, dtype=self.attn_dtype
            )
            v_planes = {"qs": v}
            v = ops.dequantize(
                v_planes, quantizer=cache_quantizer, dtype=self.attn_dtype
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if isinstance(k, ShardedTensor) and type(k) != type(q):
            k = ops.reshard_like(k, like=q)

        if isinstance(v, ShardedTensor) and type(v) != type(q):
            v = ops.reshard_like(v, like=q)

        return ops.scaled_dot_product_attention(
            q=q,  # [bs, ..., sl, dim]
            k=k,  # [bs, ..., sl, dim]
            v=v,  # [bs, ..., sl, dim]
            a=mask,  # [bs, ..., sl, sl] or None
            is_causal=mask is None,  # assumes causal masking when true
            scale=scale,  # defaults to 1/sqrt(dim)
            softcap=softcap,
            impl=attention_kernel,  # if none, automatically select a kernel
        )

    def forward_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: CacheAllocation,
        seq_block_ids: torch.Tensor,
        block_index: int,
        start_positions: torch.Tensor,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        k_quantizer: StaticScaledQuantizer = None,
        v_quantizer: StaticScaledQuantizer = None,
    ):
        # Write our one updated cache row into the cache.
        self.write_timestep(
            cache_state,
            cache_partitions=[
                unpack_to_raw_tensor(k),
                unpack_to_raw_tensor(v),
            ],
            transformer_block_index=block_index,
            seq_positions=start_positions,
            page_ids=seq_block_ids,
        )

        # Restore from the cache.
        k, v = self.read(
            cache_state,
            transformer_block_index=block_index,
            page_ids=seq_block_ids,
        )

        k = pack_raw_tensor(k, k_quantizer)
        v = pack_raw_tensor(v, v_quantizer)

        return self.attention(
            q=q,
            k=k,
            v=v,
            head_count_attn=head_count_attn,
            attention_kernel=attention_kernel,
            cache_quantizer=cache_quantizer,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            mask=mask,
        )

    def forward_prefill(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: CacheAllocation,
        seq_block_ids: torch.Tensor,
        block_index: int,
        start_positions: Optional[torch.Tensor],
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        probs_quantizer: Optional[StaticScaledQuantizer] = None,
        k_quantizer: StaticScaledQuantizer = None,
        v_quantizer: StaticScaledQuantizer = None,
    ):
        self.write(
            cache_state,
            cache_partitions=[unpack_to_raw_tensor(k), unpack_to_raw_tensor(v)],
            transformer_block_index=block_index,
            page_ids=seq_block_ids,
            start_positions=start_positions,
        )

        if start_positions is not None:
            # Restore from the cache.
            k, v = self.read(
                cache_state,
                transformer_block_index=block_index,
                page_ids=seq_block_ids,
            )

            k = pack_raw_tensor(k, k_quantizer)
            v = pack_raw_tensor(v, v_quantizer)

        return self.attention(
            q=q,
            k=k,
            v=v,
            head_count_attn=head_count_attn,
            attention_kernel=attention_kernel,
            cache_quantizer=cache_quantizer,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            mask=mask,
        )
