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
)
from sharktank import ops, kernels

__all__ = ["PagedAttention"]


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

    def shard_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(state) == 1
        if self.devices is None:
            return state

        state = ReplicatedTensor(ts=state, devices=self.devices)
        return [state]

    def unshard_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
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
        page_table = self.unflatten_page_table(state)[0]

        bs, block_seq_len, *_ = page_ids.shape
        # Blocks dim 1,2 according to the configured block stride.
        blocked_shape = [
            bs,
            block_seq_len,
            self.cache_partition_count,
            self.attn_head_count,
            self.block_seq_stride,
            self.attn_head_dim,
        ]

        # Gather both partitions and split post gather. This is more
        # computationally efficient without gather fusion:
        page_table = page_table.flatten(start_dim=0, end_dim=1)
        transformer_block_index = torch.full(
            (bs, block_seq_len), transformer_block_index, device=self.device
        )
        subblock_ids = page_ids * self.transformer_block_count + transformer_block_index
        selected = ops.index_select(page_table, 0, subblock_ids.flatten(0, 1))

        selected = selected.unflatten(0, blocked_shape[:2])
        selected = selected.transpose(3, 4)
        key = selected[:, :, 0, :].flatten(1, 2)
        value = selected[:, :, 1, :].flatten(1, 2)

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
            if page_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                page_table_as_int8 = page_table.view(dtype=torch.int8)
                part_block_as_int8 = part_block.view(dtype=torch.int8)
                page_table_as_int8.index_copy_(0, index, part_block_as_int8)
            else:
                page_table.index_copy_(0, index, part_block)

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

            if isinstance(values, ShardedTensor) and type(values) != type(page_table):
                values = ops.reshard_like(values, like=page_table)

            if page_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                page_table_as_int8 = page_table.view(dtype=torch.int8)
                values_int8 = values.view(dtype=torch.int8)
                page_table_as_int8.index_put_(indices=(index,), values=values_int8)
            else:
                page_table.index_put_(indices=(index,), values=values)


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
        caches = []
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
        cache_quantizer: Optional[QuantizerTensor],
        attention_kernel: str,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        probs_quantizer: Optional[StaticScaledQuantizer] = None,
    ):
        if self.attn_type == "gqa":
            k, v = self.gqa(head_count_attn, k, v)

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
        cache_state: List[torch.Tensor],
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
        cache_state: List[torch.Tensor],
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
