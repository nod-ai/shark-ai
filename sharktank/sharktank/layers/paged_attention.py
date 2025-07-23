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

import itertools
import torch

from sharktank.types import (
    DefaultPrimitiveTensor,
    SplitPrimitiveTensor,
    ReplicatedTensor,
    ShardedTensor,
    QuantizerTensor,
    PlanarQuantizedTensor,
    StaticScaledQuantizer,
    TensorScaledLayout,
)
from sharktank import ops, kernels
from sharktank.kernels.mlir_kernel import *
from sharktank.types.tensors import AnyTensor
from sharktank.kernels.wave.attention import wave_bhsd_masked_flash_attention, wave_prefill_attention
from iree.turbine.kernel.wave.utils.torch_utils import *
from iree.turbine.kernel.wave.templates.attention_common import *
__all__ = ["PagedAttention", "attn_type_map"]


attn_type_map = {
    "llama": "gqa",
    "grok": "gqa",
    "deepseek2": "mla",
    "llama4": "gqa",
}


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

    if isinstance(tensor, ShardedTensor) and isinstance(
        tensor.shards[0], PlanarQuantizedTensor
    ):
        return tensor.clone(ts=[t.unpack()._qs for t in tensor.shards])

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

    def allocate(self, page_count: int) -> List[torch.Tensor | ReplicatedTensor]:
        tensors = [
            torch.empty(
                [page_count, self.page_slab_flat_dims],
                dtype=self.cache_dtype,
                device=self.device,
            )
        ]

        # If we have explicit devices we should attach device information:
        if self.devices is not None:
            tensors = [ReplicatedTensor(ts=[t], devices=self.devices) for t in tensors]

        return tensors

    @property
    def state_count(self):
        return 1

    def shard_state(self, state: List[torch.Tensor]) -> List[ReplicatedTensor]:
        assert len(state) == 1
        if self.devices is None:
            return state

        state = ReplicatedTensor(ts=state, devices=self.devices)
        return [state]

    def unshard_state(
        self, state: List[torch.Tensor | ReplicatedTensor]
    ) -> List[torch.Tensor]:
        assert len(state) == 1
        state = state[0].unflatten(1, self.sub_page_dims)

        if isinstance(state, ReplicatedTensor):
            assert state.shard_count == 1
            return [state.shards[0]]
        return [state]

    def unflatten_page_table(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(state) == 1
        """Unflattens the 2D page tables to 6D tensors."""
        return [state[0].unflatten(1, self.sub_page_dims)]

    def read(
        self,
        state: List[torch.Tensor],
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor,
    ):
        # print(state)
        page_table = self.unflatten_page_table(state)[0]
        # print(page_table, page_table.shape, transformer_block_index, page_ids)
        # breakpoint()

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

        if self.devices:
            # Explicitly passing a list of one value to avoid redundant transfer inside ReplicateTensor.__init__.
            key = ReplicatedTensor(ts=[key], devices=self.devices)
            value = ReplicatedTensor(ts=[value], devices=self.devices)

        return key, value

    def write(
        self,
        *,
        state: List[torch.Tensor],
        cache_partitions: List[torch.Tensor],
        transformer_block_index: int,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        assert len(state) == 1
        assert len(cache_partitions) == self.cache_partition_count

        page_table = self.unflatten_page_table(state=state)[0]
        page_table = page_table.flatten(0, 2)

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
        state: List[torch.Tensor],
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

    def write_range(
        self,
        *,
        state: List[torch.Tensor],
        cache_partitions: List[torch.Tensor],
        transformer_block_index: int,
        seq_positions: torch.Tensor,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes a range of cache partitions to the page table.
        Similar function to `write_timestep`, but generalized for writing
        cache partitions with seq_len > 1.
        Args:
            state (List[torch.Tensor]): Current state of the KV cache allocation.
            cache_partitions (List[torch.Tensor]): K and V cache partitions.
            transformer_block_index (int): Transformer block index to write to.
            seq_positions (torch.Tensor): Positions denoting the starting index to write for a given sequence.
            page_ids (Union[torch.Tensor, ReplicatedTensor]): Page IDs to write to.
        """
        assert len(state) == 1
        assert len(cache_partitions) == self.cache_partition_count

        page_table = self.unflatten_page_table(state)[0]
        page_table = page_table.flatten(0, 4)

        device = self.device
        bs, seq_len, *_ = cache_partitions[0].shape

        if seq_len == 0:
            # If the sequence length is 0, we don't need to write anything.
            return

        positions = torch.arange(seq_len, device=device, dtype=torch.int64).unsqueeze(
            0
        ) + seq_positions.unsqueeze(
            1
        )  # [bs, seq_len]

        # Compute the logical page indices from `seq_positions`
        logical_page_index = positions // self.block_seq_stride  # [bs, seq_len]

        # Obtain the real page ids from the page table.
        real_page_ids = ops.gather(page_ids, dim=1, index=logical_page_index).view(
            bs, seq_len, 1
        )

        # Compute the page offsets within the block sequence stride.
        page_offset = (positions % self.block_seq_stride).view(bs, seq_len, 1)

        # Compute the head offsets.
        head_offset = torch.arange(self.attn_head_count, device=device).view(
            (1, 1, self.attn_head_count)
        )

        # Loop over the cache partitions and write them to the page table.
        for cache_partition_id, cache_partition in enumerate(cache_partitions):
            partitions = torch.tensor(cache_partition_id, device=device).view(1, 1, 1)

            # Compute the flat index for the page table.
            index = real_page_ids
            index = index * self.transformer_block_count + transformer_block_index
            index = index * self.cache_partition_count + partitions
            index = index * self.attn_head_count + head_offset
            index = index * self.block_seq_stride + page_offset

            # Prepare the values to write.
            values = ops.to(cache_partition, dtype=page_table.dtype)

            ops.index_put_(page_table, indices=(index,), values=values)


class ShardedCache:
    def __init__(
        self,
        *,
        shard_count: int,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        devices: List[int] | None = None,
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        cache_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        caches: list[KVCache] = []
        for i in range(shard_count):
            start = i * attn_head_count // shard_count
            end = (i + 1) * attn_head_count // shard_count
            sharded_attn_head_count = end - start

            cache = KVCache(
                transformer_block_count=transformer_block_count,
                attn_head_count=sharded_attn_head_count,
                attn_head_dim=attn_head_dim,
                cache_partition_count=cache_partition_count,
                block_seq_stride=block_seq_stride,
                cache_dtype=cache_dtype,
                device=device,
            )
            caches.append(cache)

        self.caches = caches
        self.cache_partition_count = cache_partition_count
        self.devices = devices
        self.attn_head_count = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.block_seq_stride = block_seq_stride
        self.transformer_block_count = transformer_block_count
        self.shard_count = shard_count

        self.unsharded_page_dims = [
            self.transformer_block_count,
            self.cache_partition_count,
            self.attn_head_count,
            self.block_seq_stride,
            self.attn_head_dim,
        ]

        if self.devices is None:
            self.devices = list(range(shard_count))

    @property
    def state_count(self):
        return 1

    def allocate(
        self, page_count: int, devices: List[int] | None = None
    ) -> List[SplitPrimitiveTensor]:
        assert devices is None
        shards = [cache.allocate(page_count)[0] for cache in self.caches]
        return [SplitPrimitiveTensor(ts=shards, shard_dim=1, devices=self.devices)]

    def shard_state(self, state: List[torch.Tensor]) -> List[SplitPrimitiveTensor]:
        assert len(state) == 1
        page_table = state[0].unflatten(1, self.unsharded_page_dims)

        shards = []
        head_start = 0
        for cache in self.caches:
            head_end = head_start + cache.attn_head_count
            shard = page_table[:, :, :, head_start:head_end]
            shard = shard.flatten(1)
            shards.append(shard)
            head_start = head_end

        return [SplitPrimitiveTensor(ts=shards, shard_dim=1, devices=self.devices)]

    def unshard_state(self, state: List[SplitPrimitiveTensor]) -> List[torch.Tensor]:
        assert len(state) == 1
        assert state[0].shard_count == len(self.caches)

        state = [
            cache.unshard_state([shard])[0]
            for cache, shard in zip(self.caches, state[0].shards)
        ]
        state = SplitPrimitiveTensor(ts=state, shard_dim=3, devices=self.devices)

        return [ops.unshard(state)]

    def read(
        self,
        state: List[SplitPrimitiveTensor],
        *,
        transformer_block_index: int,
        page_ids: ReplicatedTensor,
    ):
        assert len(state) == 1
        assert state[0].shard_count == self.shard_count

        for device in state[0].devices:
            assert device in page_ids.devices

        page_id_map = {d: i for i, d in enumerate(page_ids.devices)}
        page_id_shards = [page_ids.shards[page_id_map[d]] for d in state[0].devices]

        shards = []
        for shard_state, cache, shard_page_ids in zip(
            state[0].shards, self.caches, page_id_shards
        ):
            read = cache.read(
                state=[shard_state],
                transformer_block_index=transformer_block_index,
                page_ids=shard_page_ids,
            )
            shards.append(read)

        tensors = []
        for i in range(self.cache_partition_count):
            ret_shards = [s[i] for s in shards]
            tensors.append(
                SplitPrimitiveTensor(ts=ret_shards, shard_dim=2, devices=self.devices)
            )

        return tuple(tensors)

    def write(
        self,
        *,
        state: List[SplitPrimitiveTensor],
        cache_partitions: List[SplitPrimitiveTensor],
        transformer_block_index: int,
        page_ids: Union[ReplicatedTensor],
    ):
        assert len(state) == 1
        assert state[0].shard_count == self.shard_count

        for p in cache_partitions:
            assert tuple(state[0].devices) == tuple(p.devices)

        for device in state[0].devices:
            assert device in page_ids.devices

        shards = []
        for i in range(self.shard_count):
            cache_partition_shards = [p.shards[i] for p in cache_partitions]
            self.caches[i].write(
                state=[state[0].shards[i]],
                cache_partitions=cache_partition_shards,
                transformer_block_index=transformer_block_index,
                page_ids=page_ids.shards[i],
            )

    def write_timestep(
        self,
        *,
        state: List[torch.Tensor],
        cache_partitions: List[torch.Tensor],
        transformer_block_index: int,
        seq_positions: torch.Tensor,
        page_ids: torch.Tensor,
    ):
        assert len(state) == 1
        assert state[0].shard_count == self.shard_count

        shards = []
        for i in range(self.shard_count):
            cache_partition_shards = [p.shards[i] for p in cache_partitions]
            self.caches[i].write_timestep(
                state=[state[0].shards[i]],
                cache_partitions=cache_partition_shards,
                transformer_block_index=transformer_block_index,
                seq_positions=seq_positions.shards[i],
                page_ids=page_ids.shards[i],
            )

    def write_range(
        self,
        *,
        state: List[SplitPrimitiveTensor],
        cache_partitions: List[SplitPrimitiveTensor],
        transformer_block_index: int,
        seq_positions: SplitPrimitiveTensor,
        page_ids: ReplicatedTensor,
    ):
        assert len(state) == 1
        assert state[0].shard_count == self.shard_count

        for i in range(self.shard_count):
            cache_partition_shards = [p.shards[i] for p in cache_partitions]
            self.caches[i].write_range(
                state=[state[0].shards[i]],
                cache_partitions=cache_partition_shards,
                transformer_block_index=transformer_block_index,
                seq_positions=seq_positions.shards[i],
                page_ids=page_ids.shards[i],
            )


class PipelinedCache:
    def __init__(
        self,
        *,
        shard_count: int,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        cache_dtype: torch.dtype = torch.float32,
        block_to_pipeline_map: List[int],
        pipeline_to_device_map: List[List[int]],
        device: Optional[torch.device] = None,
    ):
        assert transformer_block_count == len(block_to_pipeline_map)

        pipeline_count = len(pipeline_to_device_map)

        # Determine the mapping from each unsharded transformer block to the corresponding block in the pipeline sharded cache.
        transformer_block_map = []
        pipeline_block_counts = [0] * pipeline_count
        for pipeline in block_to_pipeline_map:
            assert pipeline >= 0 and pipeline < pipeline_count
            transformer_block_map.append(pipeline_block_counts[pipeline])
            pipeline_block_counts[pipeline] += 1

        caches = []
        for pipeline in range(pipeline_count):
            devices = pipeline_to_device_map[pipeline]
            cache = build_cache(
                shard_count=shard_count,
                transformer_block_count=pipeline_block_counts[pipeline],
                attn_head_count=attn_head_count,
                attn_head_dim=attn_head_dim,
                cache_partition_count=cache_partition_count,
                block_seq_stride=block_seq_stride,
                cache_dtype=cache_dtype,
                device=device,
                devices=devices,
            )
            caches.append(cache)

        self.caches = caches
        self.pipeline_count = pipeline_count
        self.pipeline_block_counts = pipeline_block_counts
        self.transformer_block_map = transformer_block_map
        self.transformer_block_count = transformer_block_count
        self.block_to_pipeline_map = block_to_pipeline_map

        self.unsharded_page_dims = [
            transformer_block_count,
            cache_partition_count,
            block_seq_stride,
            attn_head_count,
            attn_head_dim,
        ]

    def allocate(
        self, page_count: int
    ) -> List[ReplicatedTensor | SplitPrimitiveTensor]:
        allocations = []
        for pipeline in range(self.pipeline_count):
            cache = self.caches[pipeline]
            allocation = cache.allocate(page_count)
            allocations.append(allocation)

        allocations = list(itertools.chain(*allocations))
        return allocations

    def shard_state(self, state: List[torch.Tensor]) -> List[SplitPrimitiveTensor]:
        assert len(state) == 1

        page_table = state[0].unflatten(1, self.unsharded_page_dims)

        pipelined_tensors = []
        for pipeline in range(self.pipeline_count):
            pipeline_blocks = [
                block
                for block in range(self.transformer_block_count)
                if self.block_to_pipeline_map[block] == pipeline
            ]
            tensor = ops.index_select(
                page_table, dim=1, index=torch.tensor(pipeline_blocks)
            )
            pipelined_tensors.append(tensor)

        for i, t in enumerate(pipelined_tensors):
            assert t.shape[1] == self.pipeline_block_counts[i]

        sharded = []
        for pipeline in range(self.pipeline_count):
            cache = self.caches[pipeline]
            tensor = pipelined_tensors[pipeline].flatten(1)
            subsharded = cache.shard_state([tensor])[0]
            sharded.append(subsharded)

        return sharded

    def unshard_state(self, state: List[SplitPrimitiveTensor]) -> List[torch.Tensor]:
        expected = sum([cache.state_count for cache in self.caches])
        assert len(state) == expected
        state = state.copy()

        pipelined_states = [
            [state.pop(0) for i in range(cache.state_count)] for cache in self.caches
        ]
        unsharded_states = [
            cache.unshard_state(state)
            for cache, state in zip(self.caches, pipelined_states)
        ]

        selected_tensors = []
        for block in range(self.transformer_block_count):
            pipeline = self.block_to_pipeline_map[block]
            new_block = self.transformer_block_map[block]
            current_state = unsharded_states[pipeline]

            selected = current_state[0]
            selected = selected[:, new_block].unsqueeze(1)
            selected_tensors.append(selected)

        sharded_version = SplitPrimitiveTensor(ts=selected_tensors, shard_dim=1)
        unsharded_version = ops.unshard(sharded_version).flatten(1)
        return [unsharded_version]

    @staticmethod
    def unwrap_pipelining(state):
        if not isinstance(state, list):
            if isinstance(state, ReplicatedTensor) and state.shard_count == 1:
                state = state.shards[0]
            return state

        new_state = []
        for s in state:
            if isinstance(s, ReplicatedTensor) and s.shard_count == 1:
                s = s.shards[0]
            new_state.append(s)

        return new_state

    @staticmethod
    def unwrap_like(value, state):
        if isinstance(
            value, (torch.Tensor, DefaultPrimitiveTensor, SplitPrimitiveTensor)
        ):
            return value

        src_device_map = {device: i for i, device in enumerate(value.devices)}

        target_devices = None
        for s in state:
            if isinstance(s, ReplicatedTensor):
                target_devices = s.devices
                break

        for s in state:
            if isinstance(s, SplitPrimitiveTensor):
                target_devices = s.devices
                break

        assert all(d in src_device_map for d in target_devices)

        shards = [value.shards[src_device_map[d]] for d in target_devices]
        return ReplicatedTensor(ts=shards, devices=target_devices)

    def read(
        self,
        state: List[ReplicatedTensor | SplitPrimitiveTensor],
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor | ReplicatedTensor,
    ) -> tuple[
        ReplicatedTensor | SplitPrimitiveTensor, ReplicatedTensor | SplitPrimitiveTensor
    ]:
        pipeline = self.block_to_pipeline_map[transformer_block_index]
        block = self.transformer_block_map[transformer_block_index]

        # Select the right pipeline:
        pipeline_state = [state[pipeline]]

        # Remove pipelining from the args:
        page_ids = self.unwrap_like(page_ids, pipeline_state)

        # If device pipelined we need to unwrap:
        pipeline_state = self.unwrap_pipelining(pipeline_state)
        page_ids = self.unwrap_pipelining(page_ids)

        return self.caches[pipeline].read(
            state=pipeline_state, transformer_block_index=block, page_ids=page_ids
        )

    def write(
        self,
        *,
        state: List[ReplicatedTensor | SplitPrimitiveTensor],
        cache_partitions: List[SplitPrimitiveTensor],
        transformer_block_index: int,
        page_ids: torch.Tensor | ReplicatedTensor,
    ):
        pipeline = self.block_to_pipeline_map[transformer_block_index]
        block = self.transformer_block_map[transformer_block_index]

        # Select the right pipeline:
        pipeline_state = [state[pipeline]]

        # Remove pipelining from the args:
        page_ids = self.unwrap_like(page_ids, pipeline_state)

        # If device pipelined we need to unwrap:
        pipeline_state = self.unwrap_pipelining(pipeline_state)
        page_ids = self.unwrap_pipelining(page_ids)
        cache_partitions = self.unwrap_pipelining(cache_partitions)

        return self.caches[pipeline].write(
            state=pipeline_state,
            cache_partitions=cache_partitions,
            transformer_block_index=block,
            page_ids=page_ids,
        )

    def write_timestep(
        self,
        *,
        state: List[torch.Tensor],
        cache_partitions: List[torch.Tensor],
        transformer_block_index: int,
        seq_positions: torch.Tensor,
        page_ids: torch.Tensor,
    ):
        pipeline = self.block_to_pipeline_map[transformer_block_index]
        block = self.transformer_block_map[transformer_block_index]

        # Select the right pipeline:
        pipeline_state = [state[pipeline]]

        # Remove pipelining from the args:
        page_ids = self.unwrap_like(page_ids, pipeline_state)

        # If device pipelined we need to unwrap:
        pipeline_state = self.unwrap_pipelining(pipeline_state)
        page_ids = self.unwrap_pipelining(page_ids)
        cache_partitions = self.unwrap_pipelining(cache_partitions)
        seq_positions = self.unwrap_pipelining(seq_positions)

        return self.caches[pipeline].write_timestep(
            state=pipeline_state,
            cache_partitions=cache_partitions,
            transformer_block_index=block,
            page_ids=page_ids,
            seq_positions=seq_positions,
        )

    def write_range(
        self,
        *,
        state: List[ReplicatedTensor | SplitPrimitiveTensor],
        cache_partitions: List[SplitPrimitiveTensor],
        transformer_block_index: int,
        seq_positions: SplitPrimitiveTensor,
        page_ids: ReplicatedTensor,
    ):
        pipeline = self.block_to_pipeline_map[transformer_block_index]
        block = self.transformer_block_map[transformer_block_index]

        # Select the right pipeline:
        pipeline_state = [state[pipeline]]

        # Remove pipelining from the args:
        page_ids = self.unwrap_like(page_ids, pipeline_state)

        # If device pipelined we need to unwrap:
        pipeline_state = self.unwrap_pipelining(pipeline_state)
        page_ids = self.unwrap_pipelining(page_ids)
        cache_partitions = self.unwrap_pipelining(cache_partitions)
        seq_positions = self.unwrap_pipelining(seq_positions)

        return self.caches[pipeline].write_range(
            state=pipeline_state,
            cache_partitions=cache_partitions,
            transformer_block_index=block,
            seq_positions=seq_positions,
            page_ids=page_ids,
        )


def build_cache(
    shard_count: int,
    transformer_block_count: int,
    attn_head_count: int,
    attn_head_dim: int,
    devices: List[int] | None = None,
    cache_partition_count: int = 2,
    block_seq_stride: int = 16,
    cache_dtype: torch.dtype = torch.float32,
    block_to_pipeline_map: List[int] | None = None,
    pipeline_to_device_map: List[List[int]] | None = None,
    device: Optional[torch.device] = None,
):
    if pipeline_to_device_map is not None:
        return PipelinedCache(
            shard_count=shard_count,
            transformer_block_count=transformer_block_count,
            attn_head_count=attn_head_count,
            attn_head_dim=attn_head_dim,
            cache_partition_count=cache_partition_count,
            block_seq_stride=block_seq_stride,
            cache_dtype=cache_dtype,
            device=device,
            block_to_pipeline_map=block_to_pipeline_map,
            pipeline_to_device_map=pipeline_to_device_map,
        )

    if shard_count == 1:
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

    return ShardedCache(
        shard_count=shard_count,
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
        shard_count: int = 1,
        block_to_pipeline_map: List[int] | None = None,
        pipeline_to_device_map: List[List[int]] | None = None,
    ):
        self.transformer_block_count = transformer_block_count
        self.head_count_kv = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.block_seq_stride = block_seq_stride
        self.device = device
        self.attn_dtype = attn_dtype
        self.cache_dtype = cache_dtype
        self.shard_count = shard_count
        self.attn_type = attn_type

        self.pipeline_to_device_map = pipeline_to_device_map
        if self.pipeline_to_device_map is None:
            self.pipeline_to_device_map = [list(range(shard_count))]

        self.block_to_pipeline_map = block_to_pipeline_map
        if self.block_to_pipeline_map is None:
            self.block_to_pipeline_map = [0] * self.transformer_block_count

        self.kv_cache = build_cache(
            shard_count=shard_count,
            transformer_block_count=transformer_block_count,
            attn_head_count=attn_head_count,
            attn_head_dim=attn_head_dim,
            cache_partition_count=cache_partition_count,
            block_seq_stride=block_seq_stride,
            cache_dtype=cache_dtype,
            device=device,
            block_to_pipeline_map=block_to_pipeline_map,
            pipeline_to_device_map=pipeline_to_device_map,
        )

        self.kv_cache_test = build_cache(
            shard_count=shard_count,
            transformer_block_count=transformer_block_count,
            attn_head_count=attn_head_count,
            attn_head_dim=attn_head_dim,
            cache_partition_count=cache_partition_count,
            block_seq_stride=block_seq_stride,
            cache_dtype=cache_dtype,
            device=device,
            block_to_pipeline_map=block_to_pipeline_map,
            pipeline_to_device_map=pipeline_to_device_map,
        )

    def shard_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.kv_cache.shard_state(state=state)

    def unshard_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.kv_cache.unshard_state(state=state)

    @property
    def pad_sequence_stride(self) -> int:
        return self.block_seq_stride

    def allocate(
        self, page_count: int
    ) -> List[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor]:
        return self.kv_cache.allocate(page_count=page_count)

    def read(
        self,
        state: List[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        page_ids: Optional[Union[torch.Tensor, ReplicatedTensor]] = None,
    ):
        return self.kv_cache.read(
            state=state,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
        )

    def write_timestep(
        self,
        state: List[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        cache_partitions: List[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        transformer_block_index: int,
        seq_positions: Union[torch.Tensor, ReplicatedTensor],
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        self.kv_cache.write_timestep(
            state=state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            seq_positions=seq_positions,
            page_ids=page_ids,
        )

    def write_range(
        self,
        state: List[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        cache_partitions: List[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        transformer_block_index: int,
        seq_positions: Optional[torch.Tensor],
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        self.kv_cache.write_range(
            state=state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            seq_positions=seq_positions,
            page_ids=page_ids,
        )

    def write(
        self,
        state: List[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        cache_partitions: List[torch.Tensor | SplitPrimitiveTensor | ReplicatedTensor],
        *,
        transformer_block_index: int,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        self.kv_cache.write(
            state=state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
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
        attention_kernel: str,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if attention_kernel not in ["decomposed", "sharktank", "torch", "wave"]:
            raise ValueError(
                f"Unsupported attention kernel: {attention_kernel}. "
                "Supported kernels: decomposed, sharktank, torch, wave."
            )

        if self.attn_type == "gqa":
            k, v = self.gqa(head_count_attn, k, v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if isinstance(k, ShardedTensor) and type(k) != type(q):
            k = ops.reshard_like(k, like=q)

        if isinstance(v, ShardedTensor) and type(v) != type(q):
            v = ops.reshard_like(v, like=q)

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
            attn_weights = ops.to(attn_weights, dtype=q.dtype)
            return ops.matmul(attn_weights, v)  # (bs, heads, slen, head_dim)

        elif attention_kernel == "sharktank":
            if mask is not None:
                attn_output = ops.attention_impls.masked_flash_attention(
                    q, k, v, mask[0, 0, :, :]
                )
            else:
                attn_output = kernels.flash_attention(q, k, v)
            return attn_output
        elif attention_kernel == "wave":
            if mask is None:
                # print("nicee")
                output = torch.zeros(
                    [q.shape[0], q.shape[1], q.shape[2], v.shape[3]],
                    dtype=torch.float32,
                )
                # print(q.type)
                attn_output = wave_bhsd_masked_flash_attention(q, k, v, output)
                attn_output = attn_output.to(torch.float16)
                return attn_output
            else:
                # TODO: support wave flash attention w/o is_causal mask
                pass

        # Non-decomposed
        if softcap is not None:
            raise ValueError("softcap not supported yet")

        return ops.scaled_dot_product_attention(
            q=q,  # [bs, ..., sl, dim]
            k=k,  # [bs, ..., sl, dim]
            v=v,  # [bs, ..., sl, dim]
            a=mask,  # [bs, ..., sl, sl]
            is_causal=mask is None,  # assumes causal masking when true
            scale=scale,  # defaults to 1/sqrt(dim)
            dtype=self.attn_dtype,  # apply dtype casting
        )

    def forward_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: List[torch.Tensor],
        seq_block_ids: torch.Tensor,
        block_index: int,
        start_positions: torch.Tensor,
        attention_kernel: str,
        head_count_attn: int,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        k_quantizer: StaticScaledQuantizer = None,
        v_quantizer: StaticScaledQuantizer = None,
    ):
        # print(block_index, " yeap 456")
        # raise ValueError()

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


        # print("d")
        # print("d")

        # print(start_positions)
        # raise ValueError()
        # Restore from the cache.
        k, v = self.read(
            cache_state,
            transformer_block_index=block_index,
            page_ids=seq_block_ids,
        )
        # print(k)
        # raise ValueError()

        k = pack_raw_tensor(k, k_quantizer)
        v = pack_raw_tensor(v, v_quantizer)

        return self.attention(
            q=q,
            k=k,
            v=v,
            head_count_attn=head_count_attn,
            attention_kernel=attention_kernel,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            mask=mask,
        )
    

    
    def create_inputs(self,
        shape: AttentionShape,
        dtype: torch.dtype,
    ):

        dtype = torch.float16
        N_CTX = shape.context_len
        B = shape.num_seqs
        H_KV = shape.num_kv_heads
        H_Q = shape.num_query_heads
        D = shape.head_size
        b_seq_len_prefix = to_default_device(torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32))
        if shape.fixed_seq_len_prefix:
            b_seq_len_prefix.fill_(shape.fixed_seq_len_prefix)
        b_seq_len_extend = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
        if shape.fixed_seq_len_extend:
            b_seq_len_extend.fill_(shape.fixed_seq_len_extend)
        b_seq_len = b_seq_len_prefix + b_seq_len_extend

        b_req_idx = device_arange(B, dtype=torch.int32)
        b_start_loc = device_zeros((B,), dtype=torch.int32)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = device_zeros((B,), dtype=torch.int32)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = device_zeros((B + 1,), dtype=torch.int32)
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = device_zeros((b_seq_len_prefix.sum().item(),), dtype=torch.int32)

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
            )
        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
            mean=0.1, std=0.2
        )
        v_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
            mean=0.1, std=0.2
        )

        k_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
        v_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
        q_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = device_empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype
            ).normal_(mean=0.1, std=0.2)

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        b_start_loc_extend = torch.zeros_like(b_seq_len)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = device_zeros((B + 1,), dtype=torch.int32)
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)
        logit_cap = 30.0

        b_seq_mask_len = b_seq_len_extend * b_seq_len
        # NOTE: Custom mask is of causal nature in this test. Random mask numerics
        # is not tested.
        custom_mask = device_full(
            (b_seq_mask_len.sum().item(),), fill_value=1, dtype=torch.int8
        )
        mask_offsets = device_zeros((B + 1,), dtype=torch.int32)
        mask_offsets[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
        for i in range(B):
            causal_mask = (
                torch.tril(
                    device_full(
                        (b_seq_len_extend[i], b_seq_len_extend[i]),
                        fill_value=1,
                        dtype=torch.int8,
                    ),
                    diagonal=0,
                )
                == 1
            )
            prefix_mask = device_full(
                (b_seq_len_extend[i], b_seq_len_prefix[i]), fill_value=1, dtype=torch.int8
            )
            mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
            custom_mask[mask_offsets[i] : mask_offsets[i + 1]] = mask_flatten

        max_rpe_context_length = 10
        rpe_bias = device_zeros(max_rpe_context_length + 1, dtype=torch.float32)
        rpe_bias.copy_(device_randn(max_rpe_context_length + 1, dtype=torch.float32))
        rpe_bias[max_rpe_context_length] = 0

        print(max_len_extend)

        return (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            b_req_idx,
            b_seq_len,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_offsets,
            b_start_loc,
            b_seq_len_prefix,
            extend_token_num,
            max_len_extend,
            logit_cap,
            rpe_bias,
            max_rpe_context_length,
        )
    
    def test(self,cache_state,block_index,prefix_ids):
        print("dhuifuid")

        print(self.kv_cache_test)

        print(cache_state[0].shape)

        k_cache, v_cache = self.kv_cache_test.read(
            state=cache_state,
            transformer_block_index=block_index,
            page_ids=prefix_ids,
        )
        print(k_cache, k_cache.shape, k_cache.mean())
        raise ValueError()

    def forward_prefill(
        self,
        *,
        q: torch.Tensor,            # [B, L, H_q, D]
        k: torch.Tensor,            # [B, L, H_kv, D]
        v: torch.Tensor,            # [B, L, H_kv, D]
        cache_state: List[torch.Tensor],
        seq_block_ids: torch.Tensor,  # page IDs for each block in the sequence
        block_index: int,            # transformer layer index
        attention_kernel: str,
        head_count_attn: int,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        probs_quantizer: Optional[StaticScaledQuantizer] = None,
        **_
    ) -> torch.Tensor:
        assert attention_kernel == "wave", "wave prefill only"
        if False:
            B, L, _, _ = q.shape
            chunk_size  = L // 4
            chunk_bounds = [(0, chunk_size), (chunk_size, 2*chunk_size), (2*chunk_size, 3*chunk_size), (3*chunk_size, 4*chunk_size)]

            for start, end in chunk_bounds:
                # slice this chunk
                k_c = k[:, start:end]     # [B, chunk_len, H, D]
                v_c = v[:, start:end]

                # compute intra‐page offset & page ID
                page_stride = self.block_seq_stride
                page_idx    = start // page_stride
                page_off    = start %  page_stride

                # B‑vector of start‐offsets
                seq_pos     = torch.full((B,), page_off,
                                        dtype=torch.int64, device=k.device)

                # pick exactly that one physical page column: [B,1]
                page_ids_this = seq_block_ids[:, page_idx:page_idx+1]

                # write only this sub‐range into the page
                self.kv_cache.write_range(
                    state=cache_state,
                    cache_partitions=[unpack_raw_tensor(k_c),
                                    unpack_raw_tensor(v_c)],
                    transformer_block_index=block_index,
                    seq_positions=seq_pos,
                    page_ids=page_ids_this,
                )


            return self.attention(
                q=q,
                k=k,
                v=v,
                head_count_attn=head_count_attn,
                attention_kernel=attention_kernel,
                fake_quant=fake_quant,
                softcap=softcap,
                scale=scale,
                mask=mask,
                cache_quantizer=cache_quantizer
            )
        # Configure PyTorch to print full tensors without truncation
        torch.set_printoptions(profile="full", threshold=100000, linewidth=200)

        B, L, H_q, D = q.shape
        _, _, H_kv, _ = k.shape
        device = q.device

        # Partition sequence length into two fixed-size chunks
        chunk_size = L // 2
        chunk_sizes = [chunk_size, L - chunk_size]
        # build per-chunk page IDs for prefill cache
        chunk_block_ids = (
            torch.arange(len(chunk_sizes), dtype=torch.int64, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        page_stride = self.kv_cache.block_seq_stride

        out_slices: List[torch.Tensor] = []
        offset = 0
        for local_idx, sz in enumerate(chunk_sizes):

            page_index  = offset // page_stride            # integer page ID index
            page_offset = offset % page_stride  

            # slice current chunk
            q_c = q[:, offset:offset+sz]
            k_c = k[:, offset:offset+sz]
            v_c = v[:, offset:offset+sz]

            # read previously written prefix pages
            if offset > 0:
                last_page = (offset - 1) // page_stride
                prefix_ids = seq_block_ids[:, : last_page + 1]   # shape [B, last_page+1]
                k_cache, v_cache = self.kv_cache.read(
                    state=cache_state,
                    transformer_block_index=block_index,
                    page_ids=prefix_ids,
                )
            else:
                # nothing written yet
                k_cache = torch.empty((B, 0, H_kv, D), device=device)
                v_cache = torch.empty((B, 0, H_kv, D), device=device)


            # build full pointer arrays per flattened query
            N_q_chunk = B * sz
            N_kv_cache = k_cache.shape[1]

            # 1) Number of flattened queries and prefix length:
            # 2) Build the CSR‐style “indptr” arrays:
            #    - qo_indptr[i] tells the wave kernel “the i‑th query starts at
            #      offset qo_indptr[i] in the kv_indices list”
            #    - kv_indptr[i] tells “the i‑th query writes/reads kv_indptr[i+1] – kv_indptr[i] keys”
            qo_indptr = torch.arange(
                0, N_q_chunk + 1, dtype=torch.int32, device=device
            )
            kv_indptr = qo_indptr * N_kv_cache

            # 3) Build the flat list of key‑indices:
            #    For each of the N_q_chunk queries, you want to read indices [0 … N_kv_cache-1].
            kv_indices = torch.arange(
                N_kv_cache, dtype=torch.int32, device=device
            ).unsqueeze(0)              \
            .expand(N_q_chunk, -1)     \
            .reshape(-1)               # shape [N_q_chunk * N_kv_cache]

            # compute attention
            if local_idx == 0:
                out_c = self.attention(
                    q=q_c.to(torch.float16),
                    k=k_c.to(torch.float16),
                    v=v_c.to(torch.float16),
                    head_count_attn=head_count_attn,
                    attention_kernel="wave",
                    cache_quantizer=cache_quantizer,
                    fake_quant=fake_quant,
                    softcap=softcap,
                    scale=scale,
                    mask=mask,
                    probs_quantizer=probs_quantizer,
                )
                out_c = out_c.transpose(1, 2)
            else:
                # # 1) Build per‐sequence prefix/extend lengths:
                # prefix_lengths = torch.full((B,), offset, dtype=torch.int32, device=device)
                # extend_lengths = torch.full((B,), sz,     dtype=torch.int32, device=device)

                # # 2) CSR‐style indptr of shape [B+1]:
                # kv_indptr = torch.zeros((B+1,), dtype=torch.int32, device=device)
                # kv_indptr[1:] = torch.cumsum(prefix_lengths, dim=0)

                # qo_indptr = torch.zeros((B+1,), dtype=torch.int32, device=device)
                # qo_indptr[1:] = torch.cumsum(extend_lengths, dim=0)

                # # 3) Flat list of all prefix indices:
                # total_prefix = int(prefix_lengths.sum().item())   # = B * offset
                # kv_indices   = torch.arange(
                #     total_prefix, dtype=torch.int32, device=device
                # )

                # # 4) Call your extend kernel (alias wave_prefill_attention):
                # out_flat = wave_prefill_attention(
                #     q_c.flatten(0,1).to(torch.float16),   # new‐chunk queries
                #     k_c.flatten(0,1).to(torch.float16),     # new‐chunk keys
                #     v_c.flatten(0,1).to(torch.float16),     # new‐chunk values
                #     k_cache.flatten(0,1).to(torch.float16), # prefix keys
                #     v_cache.flatten(0,1).to(torch.float16), # prefix values
                #     qo_indptr,
                #     kv_indptr,
                #     kv_indices,
                #     torch.tensor(sz, dtype=torch.int32, device=device),
                #     device_zeros(sz, H_q, D, dtype=torch.float32, device=device),
                # )
                # out_c = out_flat.view(B, sz, H_q, D)

                # 1) Build per‐sequence prefix & extend lengths
                prefix_len = offset
                extend_len = sz
                
                # CORRECT: per‑flattened‑query CSR pointers
                N_q_chunk = B * sz
                N_kv_cache = k_cache.shape[1]
           
                qo_indptr = torch.arange(
                    N_q_chunk + 1, dtype=torch.int32, device=device
                )                         # [0, 1, 2, ..., N_q_chunk]
                kv_indptr = qo_indptr * N_kv_cache
           
                # each of the N_q_chunk queries attends to the full prefix
                # so we repeat [0 .. N_kv_cache‑1] N_q_chunk times
                kv_indices = torch.arange(
                    N_kv_cache, dtype=torch.int32, device=device
                ).repeat(N_q_chunk)
                
                # 5) Now call the two‐phase kernel with truly different pointers
                out_flat = wave_prefill_attention(
                    q_c.flatten(0,1).to(dtype=torch.float16),    # new queries
                    k_c.flatten(0,1).to(dtype=torch.float16),    # new keys
                    v_c.flatten(0,1).to(dtype=torch.float16),    # new vals
                    k_cache.flatten(0,1).to(dtype=torch.float16),# prefix keys
                    v_cache.flatten(0,1).to(dtype=torch.float16),# prefix vals
                    qo_indptr,           # extend pointers
                    kv_indptr,           # prefix pointers
                    kv_indices,          # prefix indices
                    torch.tensor(sz, dtype=torch.int32, device=device),
                    device_zeros(sz, H_q, D, device=device),
                )

                out_c = out_flat.view(B, sz, H_q, D)

            seq_pos       = torch.full((B,), page_offset, dtype=torch.int64, device=device)
            page_ids_this = seq_block_ids[:, page_index : page_index + 1]  # [B,1]

            self.kv_cache.write_range(
                state=cache_state,
                cache_partitions=[unpack_raw_tensor(k_c),
                                unpack_raw_tensor(v_c)],
                transformer_block_index=block_index,
                seq_positions=seq_pos,        # intra‑page start
                page_ids=page_ids_this,       # exactly one column
            )

            out_slices.append(out_c)
            offset += sz

        # stitch outputs back to [B, L, H_q, D]
        out = torch.cat(out_slices, dim=1)
        return out


    # def forward_prefill(
    #     self,
    #     *,
    #     q: torch.Tensor,            # [B, L, H_q, D]
    #     k: torch.Tensor,            # [B, L, H_kv, D]
    #     v: torch.Tensor,            # [B, L, H_kv, D]
    #     cache_state: list[torch.Tensor],
    #     seq_block_ids: torch.Tensor,  # [B, num_blocks]
    #     block_index: int,
    #     attention_kernel: str,
    #     head_count_attn: int,
    #     cache_quantizer: Optional[QuantizerTensor],
    #     fake_quant: Optional[bool],
    #     softcap: Optional[float] = None,
    #     scale: Optional[float] = None,
    #     mask: Optional[torch.Tensor] = None,
    #     probs_quantizer: Optional[StaticScaledQuantizer] = None,
    #     **_,
    # ) -> torch.Tensor:
    #     assert attention_kernel == "wave", "wave prefill only"
    #     print("enter")

    #     B, L, H_q, D = q.shape
    #     _, _, H_kv, _ = k.shape
    #     stride = L 
    #     device = q.device

    #     out = torch.zeros((B, L, H_q, D), dtype=q.dtype, device=device)

    #     N_q    = B * L

    #     # if block_index == 0:
    #     #     # write the *prefix* KV (for block 0 that’s the first L tokens)
    #     #     self.kv_cache.write(
    #     #     state=cache_state,
    #     #     cache_partitions=[k, v],          # or torch.zeros_like(k), torch.zeros_like(v)
    #     #     transformer_block_index=0,
    #     #     page_ids=seq_block_ids,
    #     #     )


    #     if block_index == 0:
    #         # no KV written yet → pretend we have an empty cache
    #         k_cache = torch.empty((B, 0, H_kv, D),
    #                             dtype=k.dtype, device=device)
    #         v_cache = torch.empty((B, 0, H_kv, D),
    #                             dtype=v.dtype, device=device)
    #     else:
    #         prefix_page_ids = seq_block_ids[:, :block_index]   # [B, block_index]
    #         k_cache, v_cache = self.kv_cache.read(
    #             state=cache_state,
    #             transformer_block_index=block_index,
    #             page_ids=prefix_page_ids,
    #         )

    #     print(k_cache.shape, v_cache.shape)
    #     # raise ValueError()

    #     qo_indptr = torch.arange(N_q + 1, dtype=torch.int32, device=device)
    #     kv_indptr = qo_indptr * k_cache.shape[1] 
    #     kv_indices = torch.arange(k_cache.shape[1] , dtype=torch.int32, device=device) .repeat(N_q)


    #     if block_index == 0:
    #         # no past context ⇒ just do masked flash attention
    #         out = self.attention(
    #             q=q, k=k, v=v,
    #             head_count_attn=head_count_attn,
    #             attention_kernel="wave",
    #             cache_quantizer=cache_quantizer,
    #             fake_quant=fake_quant,
    #             softcap=softcap,
    #             scale=scale,
    #             mask=mask,
    #             probs_quantizer=probs_quantizer,
    #         )
    #     else:
    #         out = wave_prefill_attention(
    #             q.flatten(start_dim=0, end_dim=1).to(dtype=torch.float16),
    #             k.flatten(start_dim=0, end_dim=1).to(dtype=torch.float16),
    #             v.flatten(start_dim=0, end_dim=1).to(dtype=torch.float16),
    #             k_cache.flatten(start_dim=0, end_dim=1),
    #             v_cache.flatten(start_dim=0, end_dim=1),
    #             qo_indptr,
    #             kv_indptr,
    #             kv_indices,
    #             torch.tensor(32, dtype=torch.int32, device='cuda:0'),
    #             device_zeros(
    #                 32, 32, 128, dtype=torch.float32, device='cuda:0'
    #             )
    #         )
    #     out = out.reshape(1, stride, 32, 128)
    #     print("doned")

    #     # seq_positions = torch.zeros(B, dtype=torch.int32, device=device)   # <-- shape [B]
    #     seq_positions = torch.full(
    #         (B,),
    #         block_index * stride,
    #         dtype=torch.int32,
    #         device=device,
    #     )
        
    #     # self.kv_cache.write_range(
    #     #     state=cache_state,
    #     #     cache_partitions=[
    #     #         k.view(B, L, H_kv, D),
    #     #         v.view(B, L, H_kv, D),
    #     #     ],
    #     #     transformer_block_index=block_index,
    #     #     seq_positions=seq_positions,    # shape [B]
    #     #     page_ids=seq_block_ids,         # shape [B, num_pages]
    #     # )
    #     self.write(
    #         cache_state,
    #         cache_partitions=[unpack_raw_tensor(k), unpack_raw_tensor(v)],
    #         transformer_block_index=block_index,
    #         page_ids=seq_block_ids,
    #     )

    #     # page_id = seq_block_ids[:, block_index:block_index+1]  # shape [B,1], e.g. [[0]]
    #     # k_read, v_read = self.kv_cache.read(
    #     #     state=cache_state,
    #     #     transformer_block_index=block_index+1,
    #     #     page_ids=page_id,
    #     # )

    #     # # Now compare to the original k,v you wrote:
    #     # print("K cache correct:", torch.allclose(k_read, k))
    #     # print("V cache correct:", torch.allclose(v_read, v))
    #     return out

    # def forward_prefill(
    #     self,
    #     *,
    #     q: torch.Tensor,            # [B, L, H_q, D]
    #     k: torch.Tensor,            # [B, L, H_kv, D]
    #     v: torch.Tensor,            # [B, L, H_kv, D]
    #     cache_state: list[torch.Tensor],
    #     seq_block_ids: torch.Tensor,  # [B, num_blocks]
    #     block_index: int,
    #     attention_kernel: str,
    #     head_count_attn: int,
    #     cache_quantizer: Optional[QuantizerTensor],
    #     fake_quant: Optional[bool],
    #     softcap: Optional[float] = None,
    #     scale: Optional[float] = None,
    #     mask: Optional[torch.Tensor] = None,
    #     probs_quantizer: Optional[StaticScaledQuantizer] = None,
    #     **_,
    # ) -> torch.Tensor:
    #     assert attention_kernel == "wave", "wave prefill only"

    #     # self.test(cache_state,block_index,seq_block_ids)

    #     B, L, H_q, D = q.shape
    #     _, _, H_kv, _ = k.shape
    #     stride = L #self.kv_cache.block_seq_stride
    #     device = q.device

    #     out = torch.zeros((B, L, H_q, D), dtype=q.dtype, device=device)

    #     print("sublime")
    #     for start in range(0, L, stride):
    #         # if start == 0:
    #         #     out = self.attention(
    #         #         q=q,
    #         #         k=k,
    #         #         v=v,
    #         #         head_count_attn=head_count_attn,
    #         #         attention_kernel=attention_kernel,
    #         #         cache_quantizer=cache_quantizer,
    #         #         fake_quant=fake_quant,
    #         #         softcap=softcap,
    #         #         scale=scale,
    #         #         mask=mask,
    #         #         probs_quantizer=probs_quantizer,
    #         #     )
    #         #     continue
    #         print("giggity", start, L)
    #         end       = min(start + stride, L)
    #         chunk_len = end - start

    #         # slice and flatten the new chunk
    #         q_chunk = q[:, start:end]    # [B, chunk_len, H_q, D]
    #         k_chunk = k[:, start:end]    # [B, chunk_len, H_kv, D]
    #         v_chunk = v[:, start:end]    # [B, chunk_len, H_kv, D]
    #         N_q    = B * chunk_len
    #         qf     = q_chunk.reshape(N_q, H_q, D)
    #         # print(qf)
    #         # breakpoint()
    #         kf_new = k_chunk.reshape(N_q, H_kv, D)
    #         vf_new = v_chunk.reshape(N_q, H_kv, D)

    #         # write into paged cache (if you still need it; can be no-op)
    #         # seq_positions = torch.arange(start, end, device=device)
    #         # self.kv_cache.write_range(...)

    #         # read *all* prefix back—but we will ignore it
    #         num_blocks = (end + stride - 1) // stride
    #         prefix_ids = seq_block_ids[:, :num_blocks]
    #         k_cache, v_cache = self.kv_cache.read(
    #             state=cache_state,
    #             transformer_block_index=block_index,
    #             page_ids=prefix_ids,
    #         )
    #         print(k_cache, k_cache.shape, "slrue")
    #         # if isinstance(k_cache, ReplicatedTensor):
    #         #     k_cache = k_cache.shards[0]
    #         #     v_cache = v_cache.shards[0]

    #         N_cache = k_cache.shape[1]
    #         N_kv    = B * N_cache

    #         # ** REPLACE** the real cache with zeros of the same flattened shape:
    #         # kf_cache = torch.zeros((N_kv, H_kv, D), dtype=q.dtype, device=device)
    #         # vf_cache = torch.zeros((N_kv, H_kv, D), dtype=q.dtype, device=device)

    #         # qo_indptr  = torch.arange(N_q+1,  device=device, dtype=torch.int32) * chunk_len * 0
    #         # kv_indptr  = torch.arange(N_kv+1, device=device, dtype=torch.int32) * N_cache * 0
    #         # kv_indices = torch.arange(N_cache, device=device, dtype=torch.int32).repeat(N_kv)

    #         qo_indptr = torch.arange(N_q + 1,
    #                                 dtype=torch.int32,
    #                                 device=device)

    #         # (2) A length-(N_q+1) pointer into the *kv_indices* array,
    #         #     each slot i saying “for query i, you’ll consume these many KV indices”:
    #         kv_indptr = qo_indptr * N_cache   # → [0, N_cache, 2*N_cache, …, N_q*N_cache]

    #         # (3) A flattened list of per‑query KV‑buffer offsets:
    #         #     for each of the N_q queries you repeat [0,1,2,…,N_cache-1]
    #         kv_indices = torch.arange(N_cache, dtype=torch.int32, device=device) .repeat(N_q)

    #         tot_new    = torch.tensor(4096 , device=device, dtype=torch.int32) # 

    #         buf = torch.zeros((N_q, chunk_len, D), dtype=torch.float32, device=device)
    #         print("A", tot_new, buf.shape)
    #         # dispatch

    #         qf = qf.to(dtype=torch.float16)
    #         kf_new = kf_new.to(dtype=torch.float16)
    #         vf_new = vf_new.to(dtype=torch.float16)
    #         kf_cache = k_cache[0].to(dtype=torch.float16)
    #         vf_cache = v_cache[0].to(dtype=torch.float16)

    #         q_extend = qf
    #         k_extend = kf_new
    #         v_extend = vf_new
    #         k_buffer = kf_cache
    #         v_buffer = vf_cache
    #         max_len_extend = 32

    #         del qf
    #         del kf_new
    #         del vf_new
    #         del kf_cache
    #         del vf_cache


    #         DEBUG = False
    #         # if DEBUG:
    #         #     shape = AttentionShape(
    #         #         context_len=4,
    #         #         batch_size=1,
    #         #         num_query_heads=32,
    #         #         num_kv_heads=8,
    #         #         query_seq_len=32,
    #         #         head_size_kv=8,
    #         #         head_size=128,
    #         #         kv_seq_len=32,
    #         #         num_seqs=32,
    #         #     )
    #         #     (
    #         #         _, #q_extend,
    #         #         _, #k_extend,
    #         #         _, #v_extend,
    #         #         _, #k_buffer,
    #         #         _, #v_buffer,
    #         #         b_req_idx,
    #         #         b_seq_len,
    #         #         qo_indptr,
    #         #         kv_indptr,
    #         #         kv_indices,
    #         #         custom_mask,
    #         #         mask_offsets,
    #         #         b_start_loc,
    #         #         b_seq_len_prefix,
    #         #         extend_token_num,
    #         #         max_len_extend,
    #         #         logit_cap,
    #         #         _,
    #         #         _,
    #         #     ) = self.create_inputs(shape, torch.float16)

    #         #     # flat = wave_prefill_attention(
    #         #     #     qf,       kf_new,    vf_new,
    #         #     #     kf_cache, vf_cache,
    #         #     #     qo_indptr, kv_indptr, kv_indices,
    #         #     #     max_seq_len=tot_new,   c=buf,
    #         #     # )
    #         #     print("B")

    #         #     l = [
    #         #         q_extend,
    #         #         k_extend,
    #         #         v_extend,
    #         #         k_buffer,
    #         #         v_buffer,
    #         #         qo_indptr,
    #         #         kv_indptr,
    #         #         kv_indices,
    #         #     ]
    #         #     for ll in l:
    #         #         if len(ll.shape) > 1:
    #         #             print(ll.shape)
    #         #         else:
    #         #             print(ll.shape, ll)
    #         #     max_len_extend = 3200
    #         #     print(max_len_extend)
    #         #     # breakpoint()
    #         #     # reshape back
    #         #     # out[:, start:end, :, :] = flat.reshape(B, chunk_len, H_q, D)

    #         max_len_extend = 1

    #         print(qo_indptr.shape)
    #         print(kv_indptr.shape)
    #         print(kv_indices.shape)
    #         # breakpoint()

    #         # qo_indptr = torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    #         #     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    #         #     device='cuda:0', dtype=torch.int32)
            
    #         # kv_indptr = torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    #         #     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    #         #     device='cuda:0', dtype=torch.int32)

    #         qo_indptr = torch.cat([qo_indptr, qo_indptr[-1:]], dim=0)
    #         kv_indptr = torch.cat([kv_indptr, kv_indptr[-1:]], dim=0)
            
    #         # kv_indices = torch.tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
    #         #     36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62],
    #         #     device='cuda:0', dtype=torch.int32)
            
    #         print(q_extend.shape, q_extend.dtype, q_extend.device, "is q_extend.shape")
    #         print(k_extend.shape, k_extend.dtype, k_extend.device, "is k_extend.shape")
    #         print(v_extend.shape, v_extend.dtype, v_extend.device,  "is v_extend.shape")
    #         print(k_buffer.shape, k_buffer.dtype, k_buffer.device,  "is k_buffer.shape")
    #         print(v_buffer.shape, v_buffer.dtype, v_buffer.device,  "is v_buffer.shape")
    #         print(buf.shape, buf.dtype, buf.device,  "is buf.shape")
    #         print(q_extend, qo_indptr.shape, kv_indptr.shape, kv_indices.shape)
    #         print(qo_indptr.device)
    #         print(kv_indptr.device)
    #         print(kv_indices.device)
    #         # breakpoint()

    #         print("k_buffer contiguous?", k_buffer.is_contiguous(), "strides:", k_buffer.stride())
    #         print("v_buffer contiguous?", v_buffer.is_contiguous(), "strides:", v_buffer.stride())


    #         # k_buffer = k_buffer.clone()
    #         # v_buffer = v_buffer.clone()
    #         print("nextarinoes")
    #         print("k_buffer contiguous?", k_buffer.is_contiguous(), "strides:", k_buffer.stride())
    #         print("v_buffer contiguous?", v_buffer.is_contiguous(), "strides:", v_buffer.stride())
            

    #         # k_buffer = torch.empty((0, H_kv, D), device=k_buffer.device, dtype=k_buffer.dtype)
    #         # v_buffer = torch.empty((0, H_kv, D), device=v_buffer.device, dtype=k_buffer.dtype)

    #         out = wave_prefill_attention(
    #             q_extend,
    #             k_extend,
    #             v_extend,
    #             k_buffer,
    #             v_buffer,
    #             qo_indptr,
    #             kv_indptr,
    #             kv_indices,
    #             torch.tensor(32, dtype=torch.int32, device='cuda:0'),
    #             device_zeros(
    #                 32, 32, 128, dtype=torch.float32, device='cuda:0'
    #             )
    #         )
    #         out = out.reshape(1, stride, 32, 128)

    #     print(out.shape, " hella cool man")
    #     print(out)
    #     # raise ValueError()

    #     # raise ValueError()

    #     return out