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

from itertools import accumulate
from typing import Optional, Tuple, Union, List

import abc
import math

import torch

from sharktank.types import (
    SplitPrimitiveTensor,
    ReplicatedTensor,
    QuantizerTensor,
    PlanarQuantizedTensor,
    StaticScaledQuantizer,
)
from sharktank import ops, kernels
from sharktank.kernels.mlir_kernel import *

__all__ = ["PagedAttention"]

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

    SOURCE_TY = Dtype.SOURCE_TY
    I64 = Dtype.I64

    @mlir_kernel(
        inputs=(
            MLIRTensor[
                CACHE_SIZE,
                T_BLOCK,
                PART,
                BLOCK_SEQ_STRIDE,
                HEAD_COUNT_KV,
                ATTN_HEAD_DIM,
                SOURCE_TY,
            ],
            MLIRTensor[BATCH, PAGES, I64],
            MLIRTensor[I64],
            MLIRTensor[I64],
        ),
        results=(
            MLIRTensor[
                BATCH, PAGES, BLOCK_SEQ_STRIDE, HEAD_COUNT_KV, ATTN_HEAD_DIM, SOURCE_TY
            ],
        ),
    )
    def paged_attention_kv_cache_gather(
        source, page_ids, transformer_idx, partition_idx, result
    ):
        # We generate the tensor.extract version for now, but once we have
        # iree_linalg_ext.gather, we should be generating that instead.
        mlir = """
        module {
        util.func @{{kernel_name}}(%source: !source,
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

          %batches = tensor.dim %page_ids, %c0 : !page_ids
          %pages = tensor.dim %page_ids, %c1 : !page_ids
          %empty = tensor.empty(%batches, %pages) : !result
          %result = linalg.generic {
            indexing_maps = [
            affine_map<(b, p, stride, head_count, head_dim) -> (b, p)>,
            affine_map<(b, p, stride, head_count, head_dim) -> (b, p, stride, head_count, head_dim)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
            ins(%page_ids : !page_ids)
            outs(%empty : !result) {
            ^bb0(%in: !page_ids_dtype, %o: !source_dtype):
              %p = arith.index_cast %in : !page_ids_dtype to index
              %stride = linalg.index 2 : index
              %head_count = linalg.index 3 : index
              %head_dim = linalg.index 4 : index
              %extracted = tensor.extract %source[%p, %t_id, %p_id, %stride, %head_count, %head_dim] : !source
              linalg.yield %extracted : !source_dtype
            } -> !result
            util.return %result : !result
        }
        }
        """
        return MLIRSpec(mlir)

    return paged_attention_kv_cache_gather


kv_cache_gather = KVCacheGatherKernel()

# Paged Attention Implementation


class PagedAttention:
    """Implementation of paged attention

    The page table slab is physically represented as a 2D tensor:
        [page_count, flattened_dims]

    Each "page" can be thought of as a 6D view onto:

    * transformer block
    * cache partition (K or V cache)
    * block sequence stride (number of sequence positions per block)
    * attention heads
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
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        cache_dtype: torch.dtype = torch.float32,
        attn_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        shard_count: int = 1,
        block_to_pipeline_map: list[int] | None = None,
        pipeline_to_device_map: list[list[int]] | None = None,
    ):
        self.transformer_block_count = transformer_block_count
        self.head_count_kv = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride
        self.shard_count = shard_count

        self.block_to_pipeline_map = (
            [0] * transformer_block_count
            if block_to_pipeline_map is None
            else block_to_pipeline_map
        )
        assert all(
            a <= b
            for a, b in zip(self.block_to_pipeline_map, self.block_to_pipeline_map[1:])
        )

        self.pipeline_to_device_map = (
            [list(range(self.shard_count))]
            if pipeline_to_device_map is None
            else pipeline_to_device_map
        )
        self.pipeline_count = len(self.pipeline_to_device_map)

        if attn_head_count % shard_count != 0:
            raise ValueError(
                f"The attention head count {attn_head_count} must be a multiple of the tensor parallelism size {shard_count}."
            )

        self.pipeline_to_block_count = list(
            sum(1 for block in self.block_to_pipeline_map if block == i)
            for i in range(self.pipeline_count)
        )
        self.pipeline_to_block_offset = [0] + list(
            accumulate(self.pipeline_to_block_count)
        )[:-1]

        # Some derived values based on attributes.
        self.sub_page_dims = [
            [
                self.pipeline_to_block_count[pipeline],
                self.cache_partition_count,
                self.block_seq_stride,
                self.head_count_kv // self.shard_count,
                self.attn_head_dim,
            ]
            for pipeline in range(self.pipeline_count)
        ]
        self.page_slab_flat_dims = [
            math.prod(sub_page_dim) for sub_page_dim in self.sub_page_dims
        ]

        self.device = device
        self.cache_dtype = cache_dtype
        self.attn_dtype = attn_dtype

    def unflatten_page_tables(
        self, state: list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor]
    ) -> list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor]:
        """Unflattens the 2D page tables to 6D tensors."""
        assert (
            len(state) == self.pipeline_count
        ), f"Expected {self.pipeline_count}-element state. Got: {len(state)}"

        unflattened = []
        for pipeline, page_slab in enumerate(state):
            shards = [page_slab] if self.shard_count == 1 else page_slab.shards
            shards = [
                shard.unflatten(1, self.sub_page_dims[pipeline]) for shard in shards
            ]

            result = shards[0]
            if self.shard_count > 1:
                result = SplitPrimitiveTensor(
                    ts=shards, shard_dim=4, devices=page_slab.devices
                )
            elif pipeline > 1:
                result = ReplicatedTensor(ts=shards, devices=page_slab.devices)

            unflattened.append(result)

        return unflattened

    def shard_state(
        self, state: List[torch.Tensor]
    ) -> List[torch.Tensor | SplitPrimitiveTensor]:
        """Shard an unsharded state.
        We can't just split the slab on the sub page dims.
        First it needs to be reinterpreted into the actual shape.
        The split the head dimension, then flatten each shard.
        This is a work-around for the lack of block-cyclic sharded tensor type."""
        if self.shard_count == 1 and self.pipeline_count == 1:
            return state

        page_table = state[0].reshape(
            [
                -1,
                self.transformer_block_count,
                self.cache_partition_count,
                self.block_seq_stride,
                self.head_count_kv,
                self.attn_head_dim,
            ]
        )

        flat_sharded_page_tables = []
        for pipeline in range(self.pipeline_count):
            devices = self.pipeline_to_device_map[pipeline]

            block_min = self.pipeline_to_block_offset[pipeline]
            block_sz = self.pipeline_to_block_count[pipeline]
            block_max = block_min + block_sz
            selected = page_table[:, block_min:block_max, ...]

            if self.shard_count == 1:
                sharded_page_table = ops.replicate(selected, count=1, devices=devices)
            else:
                sharded_page_table = ops.reshard_split(
                    selected, dim=4, count=self.shard_count, devices=devices
                )

            shards_flattened = [
                ops.flatten(shard, start_dim=1) for shard in sharded_page_table.shards
            ]
            flat_sharded_page_tables.append(
                SplitPrimitiveTensor(ts=shards_flattened, shard_dim=1, devices=devices)
                if self.shard_count > 1
                else ReplicatedTensor(ts=shards_flattened, devices=devices)
            )
        return flat_sharded_page_tables

    def unshard_state(
        self, state: List[torch.Tensor | SplitPrimitiveTensor]
    ) -> List[torch.Tensor]:
        state = self.unflatten_page_tables(state)
        state = [ops.unshard(s) for s in state]
        catted = ops.cat(state, dim=1)
        catted = ops.flatten(catted, 1)
        return catted

    @property
    def pad_sequence_stride(self) -> int:
        return self.block_seq_stride

    def allocate(
        self, page_count: int
    ) -> list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor]:
        """Allocates tensor state for a page table for the given capacity in
        pages.
        """
        shards = [
            [
                torch.empty(
                    [page_count, shard_dims],
                    dtype=self.cache_dtype,
                    device=self.device,
                )
                for _ in range(self.shard_count)
            ]
            for shard_dims in self.page_slab_flat_dims
        ]

        if self.shard_count == 1 and self.pipeline_count == 1:
            return shards[0]

        return [
            (
                SplitPrimitiveTensor(ts=shards[i], shard_dim=1, devices=devices)
                if len(shards[i]) > 1
                else ReplicatedTensor(ts=shards[i], devices=devices)
            )
            for i, devices in enumerate(self.pipeline_to_device_map)
        ]

    def read(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        page_ids: Optional[Union[torch.Tensor, ReplicatedTensor]] = None,
    ):
        """Reads K/V caches the page table for the given page_ids.

        Args:
        state: State struct as returned from allocate().
        transformer_block_index: The index of the transformer block accessing
            the cache.
        page_ids: Tensor of [bs, max_seqlen // block_pos_stride] of page ids
            to access.

        Returns the K/V cache partitions, linearized. Note that this reference
        approach to reading by materializing linearly may not be terribly
        efficient unless if the compiler can fuse the gather.
        """
        page_tables = self.unflatten_page_tables(state)  # 6D
        pipeline = self.block_to_pipeline_map[transformer_block_index]
        page_table = page_tables[pipeline]
        block_offset = self.pipeline_to_block_offset[pipeline]
        transformer_block_index = transformer_block_index - block_offset

        # TODO: mlir_kernel doesn't support non-tensor args yet, so use 0-D
        # tensors instead.
        t_id = torch.tensor(transformer_block_index, dtype=torch.int64)
        key_p_id = torch.tensor(0, dtype=torch.int64)
        value_p_id = torch.tensor(1, dtype=torch.int64)

        key = kv_cache_gather(page_table, page_ids, t_id, key_p_id)
        value = kv_cache_gather(page_table, page_ids, t_id, value_p_id)

        key = key.flatten(1, 2)
        value = value.flatten(1, 2)

        return key, value

    def write_timestep(
        self,
        state: list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        # List of [bs, 1, attn_head_count, attn_head_dim]
        cache_partitions: list[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        *,
        transformer_block_index: int,
        # [bs]
        seq_positions: Union[torch.Tensor, ReplicatedTensor],
        # [bs, max_seqlen // block_pos_stride]
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes a single batched timestep across all cache partitions.

        Note that this internally loops over the batch size, which cannot be
        dynamic.
        """
        device = self.device
        page_tables = self.unflatten_page_tables(state)  # 6D
        pipeline = self.block_to_pipeline_map[transformer_block_index]
        devices = self.pipeline_to_device_map[pipeline]
        transformer_block_count = self.pipeline_to_block_count[pipeline]
        block_offset = self.pipeline_to_block_offset[pipeline]

        page_table = page_tables[pipeline]
        page_table = page_table.flatten(0, 3)
        bs, *_ = seq_positions.shape
        assert len(cache_partitions) == self.cache_partition_count

        transformer_block_index = transformer_block_index - block_offset

        # [bs, 1, atten_head_count, attn_head_dim]
        for idx, cache_partition in enumerate(cache_partitions):
            # [bs, 1]
            page_index = seq_positions // self.block_seq_stride

            page_id = ops.gather(page_ids, dim=1, index=page_index.unsqueeze(1))
            page_offset = (seq_positions % self.block_seq_stride).unsqueeze(1)

            # [1, 1]
            partitions = torch.tensor(idx, device=device).unsqueeze(0)
            transformer_block = torch.full(
                (bs, 1), transformer_block_index, device=device
            )
            if isinstance(seq_positions, ReplicatedTensor):
                partitions = [partitions] * seq_positions.shard_count
                transformer_block = [transformer_block] * seq_positions.shard_count

                partitions = ReplicatedTensor(ts=partitions, devices=devices)
                transformer_block = ReplicatedTensor(
                    ts=transformer_block, devices=devices
                )

            partitions = partitions.repeat(bs, 1)

            index = page_id
            index = index * transformer_block_count + transformer_block
            index = index * self.cache_partition_count + partitions
            index = index * self.block_seq_stride + page_offset

            values = ops.to(cache_partition, dtype=page_table.dtype)
            if page_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                page_table_as_int8 = page_table.view(dtype=torch.int8)
                values_int8 = values.view(dtype=torch.int8)
                page_table_as_int8.index_put_(indices=(index,), values=values_int8)
            else:
                page_table.index_put_(indices=(index,), values=values)

        return

    def write(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        cache_partitions: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        page_tables = self.unflatten_page_tables(state)  # 6D
        page_table = page_tables[self.block_to_pipeline_map[transformer_block_index]]
        bs, block_seq_len, *_ = page_ids.shape

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        pipeline = self.block_to_pipeline_map[transformer_block_index]

        # We have to offset according to the pipeline sharding:
        block_offset = self.pipeline_to_block_offset[pipeline]
        transformer_block_index = transformer_block_index - block_offset

        transformer_block_count = self.pipeline_to_block_count[pipeline]
        page_stride = transformer_block_count * self.cache_partition_count
        transformer_block_stride = self.cache_partition_count
        base_subblock_ids = page_ids * page_stride + (
            transformer_block_index * transformer_block_stride
        )

        for index, partition in enumerate(cache_partitions):
            part_block_view = partition.unflatten(
                1, (block_seq_len, self.block_seq_stride)
            )
            part_block_view = part_block_view.flatten(0, 1)

            subblock_ids = (
                (base_subblock_ids + index) if index > 0 else base_subblock_ids
            ).flatten(0, 1)

            part_block = ops.to(part_block_view, dtype=subblock_table.dtype)
            if subblock_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                subblock_table_as_int8 = subblock_table.view(dtype=torch.int8)
                part_block_as_int8 = part_block.view(dtype=torch.int8)
                subblock_table_as_int8.index_copy_(0, subblock_ids, part_block_as_int8)
            else:
                subblock_table.index_copy_(0, subblock_ids, part_block)

    def attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_count_attn: int,
        attention_kernel: str,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        probs_quantizer: Optional[StaticScaledQuantizer] = None,
    ):
        gqa_n_rep = head_count_attn // self.head_count_kv
        assert gqa_n_rep > 0
        if gqa_n_rep > 1:

            def repeat_kv(x: torch.Tensor) -> torch.Tensor:
                bs, slen, n_kv_heads, head_dim = x.shape
                unsq = x.unsqueeze(-2)
                exp = ops.expand(unsq, (bs, slen, n_kv_heads, gqa_n_rep, head_dim))
                return exp.flatten(2, 3)

            k = repeat_kv(k)
            v = repeat_kv(v)

        # Fake quant is already dequantized when stored in the cache.
        if cache_quantizer and not fake_quant:
            k = cache_quantizer.dequantize_raw_tensor(k, self.attn_dtype, name="xk_deq")
            v = cache_quantizer.dequantize_raw_tensor(v, self.attn_dtype, name="xv_deq")

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = ops.to(q, dtype=self.attn_dtype)
        k = ops.to(k, dtype=self.attn_dtype)
        v = ops.to(v, dtype=self.attn_dtype)
        if mask is not None:
            mask = ops.to(mask, dtype=self.attn_dtype)

        # Decomposed
        if attention_kernel == "decomposed":
            if isinstance(q, PlanarQuantizedTensor):
                q = q.unpack().dequantize()
            if isinstance(k, PlanarQuantizedTensor):
                k = k.unpack().dequantize()
            if isinstance(v, PlanarQuantizedTensor):
                v = v.unpack().dequantize()

            attn_weights = ops.matmul(
                q.to(torch.float32), k.transpose(2, 3).to(torch.float32)
            )
            attn_weights = attn_weights / math.sqrt(self.attn_head_dim)

            # Flash attention.
            if softcap is not None:
                attn_weights = softcap * torch.tanh(attn_weights / softcap)

            # Apply attention mask.
            if mask is None:
                mask = torch.full(
                    (attn_weights.shape[2], attn_weights.shape[3]), float("-inf")
                )
                mask = torch.triu(mask, diagonal=1)[None, None, :, :]
                attn_weights = attn_weights + mask
            else:
                attn_weights = attn_weights + mask

            attn_weights = ops.softmax(
                ops.to(attn_weights, dtype=torch.float32), dim=-1
            )
            if probs_quantizer is not None:
                if fake_quant:
                    attn_weights = (
                        probs_quantizer.quantize(attn_weights).unpack().dequant()
                    )
                else:
                    attn_weights = probs_quantizer.quantize(attn_weights).unpack().qs
            attn_weights = ops.to(attn_weights, dtype=q.dtype)
            return ops.matmul(attn_weights, v)  # (bs, heads, slen, head_dim)
        elif attention_kernel == "sharktank":
            if mask is not None:
                attn_output = kernels.masked_flash_attention(
                    q,
                    k,
                    v,
                    mask[0, 0, :, :],
                    torch.tensor(1 / math.sqrt(self.attn_head_dim)),
                )
            else:
                attn_output = kernels.flash_attention(q, k, v)
            return attn_output
        else:
            # Non-decomposed
            if softcap is not None:
                raise ValueError("softcap not supported yet")

            return ops.scaled_dot_product_attention(
                q=q,  # [bs, ..., sl, dim]
                k=k,  # [bs, ..., sl, dim]
                v=v,  # [bs, ..., sl, dim]
                a=mask,  # [bs, ..., sl, sl]
                is_causal=mask is None,  # assumes causal masking when true
                scale=None,  # defaults to 1/sqrt(dim)
            )

    def forward_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: list[torch.Tensor],
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
    ):
        # Write our one updated cache row into the cache.
        self.write_timestep(
            cache_state,
            cache_partitions=[
                k,
                v,
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
        cache_state: list[torch.Tensor],
        seq_block_ids: torch.Tensor,
        block_index: int,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        probs_quantizer: Optional[StaticScaledQuantizer] = None,
    ):
        self.write(
            cache_state,
            cache_partitions=[k, v],
            transformer_block_index=block_index,
            page_ids=seq_block_ids,
        )

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
            probs_quantizer=probs_quantizer,
        )
