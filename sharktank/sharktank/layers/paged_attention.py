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

from abc import ABC, abstractmethod
from typing import Optional, Union, List
from collections import defaultdict
import math

import torch

from amdsharktank.layers.configs.llm_configs import LlamaModelConfig, ParallelismConfig
from amdsharktank.layers.kv_cache import KVCache, CacheAllocation
from amdsharktank.types import (
    AnyTensor,
    DefaultPrimitiveTensor,
    QuantizerTensor,
    PlanarQuantizedTensor,
    ShardedTensor,
    StaticScaledQuantizer,
    TensorScaledLayout,
    QuantizedTensor,
    ReplicatedTensor,
    unpack_to_raw_tensor,
    pack_raw_tensor,
)
from amdsharktank import ops
from amdsharktank.utils.attention import *
from amdsharktank.kernels.mlir_kernel import *

__all__ = ["PagedAttention", "PagedKVCache", "attn_type_map"]

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


class PagedKVCache(KVCache, ABC):
    @property
    @abstractmethod
    def block_size_elements_per_device(self) -> list[int]:
        ...


class DefaultPagedKVCache(PagedKVCache):
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
    ):
        self.transformer_block_count = transformer_block_count
        self.attn_head_count = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride
        self._cache_dtype = cache_dtype
        self.device = device

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
    def cache_dtype(self) -> torch.dtype:
        return self._cache_dtype

    @property
    def block_size_elements_per_device(self) -> list[int]:
        return [self.page_slab_flat_dims]

    @property
    def state_count(self) -> int:
        return 1

    def unflatten_page_table(self, state: CacheAllocation) -> torch.Tensor:
        """Unflattens the 2D page tables to 6D tensors."""
        assert len(state) == 1
        return state[0].unflatten(1, self.sub_page_dims)

    def read(
        self,
        state: CacheAllocation,
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor,
        k_quantizer: StaticScaledQuantizer | None = None,
        v_quantizer: StaticScaledQuantizer | None = None,
    ) -> torch.Tensor | QuantizedTensor:
        page_table = self.unflatten_page_table(state)

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

        key = pack_raw_tensor(key, k_quantizer, dtype=torch.float16)
        value = pack_raw_tensor(value, v_quantizer, dtype=torch.float16)

        return key, value

    def write(
        self,
        state: CacheAllocation,
        *,
        cache_partitions: List[torch.Tensor | QuantizedTensor],
        transformer_block_index: int,
        page_ids: torch.Tensor,
        start_positions: torch.Tensor | None,
    ) -> None:
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        assert len(state) == 1
        assert len(cache_partitions) == self.cache_partition_count
        cache_partitions = [unpack_to_raw_tensor(cp) for cp in cache_partitions]

        page_table = self.unflatten_page_table(state=state)
        page_table = page_table.flatten(0, 2)

        block_seq_len = cache_partitions[0].shape[1] // self.block_seq_stride

        if start_positions is not None:
            page_index = (
                start_positions.unsqueeze(1) // self.block_seq_stride
            ) + ops.arange(block_seq_len)
            page_ids = ops.gather(page_ids, dim=1, index=page_index)

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
        state: CacheAllocation,
        *,
        cache_partitions: List[torch.Tensor | QuantizedTensor],
        transformer_block_index: int,
        seq_positions: torch.Tensor,
        page_ids: torch.Tensor,
    ) -> None:
        assert len(state) == 1
        assert len(cache_partitions) == self.cache_partition_count
        cache_partitions = [unpack_to_raw_tensor(cp) for cp in cache_partitions]

        page_table = self.unflatten_page_table(state)
        page_table = page_table.flatten(0, 4)

        device = self.device
        bs, *_ = seq_positions.shape

        page_index = seq_positions // self.block_seq_stride
        page_index = page_index.unsqueeze(1)
        page_id = ops.gather(page_ids, dim=1, index=page_index).view((bs, 1, 1))
        page_offset = (seq_positions % self.block_seq_stride).view((bs, 1, 1))
        head_offset = ops.arange(self.attn_head_count, device=device).view(
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


class PipelinedPagedKVCache(PagedKVCache):
    def __init__(
        self,
        *,
        parallelism_config: ParallelismConfig,
        **sub_kwargs,
    ):
        self.config = parallelism_config
        self.block_seq_stride = sub_kwargs.get("block_seq_stride")
        self.attn_head_count = sub_kwargs.get("attn_head_count")

        self.kv_caches: list[DefaultPagedKVCache] = []
        for num_blocks in self.config.num_blocks_per_pipeline:
            sub_kwargs["transformer_block_count"] = num_blocks
            self.kv_caches.append(DefaultPagedKVCache(**sub_kwargs))

    def allocate(self, page_count: int) -> CacheAllocation:
        allocations = []
        for kv_cache in self.kv_caches:
            allocations.extend(kv_cache.allocate(page_count=page_count))
        return CacheAllocation(allocations)

    @property
    def cache_dtype(self) -> torch.dtype:
        return self.kv_caches[0].cache_dtype

    @property
    def block_size_elements_per_device(self) -> list[int]:
        elements = []
        for cache in self.kv_caches:
            elements.extend(cache.block_size_elements_per_device)
        return elements

    @property
    def state_count(self) -> int:
        return len(self.kv_caches)

    def adjust_index(self, index: int) -> int:
        offset = self.config.first_block_in_pipeline_for_block(index)
        return index - offset

    def read(
        self,
        state: CacheAllocation,
        *,
        transformer_block_index: int,
        page_ids: ReplicatedTensor,
        k_quantizer: ReplicatedTensor | None = None,
        v_quantizer: ReplicatedTensor | None = None,
    ) -> Union[torch.Tensor, QuantizedTensor]:
        pipeline = self.config.pipeline_for_block(transformer_block_index)
        transformer_block_index = self.adjust_index(transformer_block_index)

        assert len(page_ids.shards) == 1, "Tensor parallelism not supported."
        if k_quantizer is not None:
            assert len(k_quantizer.shards) == 1, "Tensor parallelism not supported."
        if v_quantizer is not None:
            assert len(v_quantizer.shards) == 1, "Tensor parallelism not supported."

        state = CacheAllocation([state[pipeline]])
        page_ids = page_ids.shards[0]
        k_quantizer = k_quantizer.shards[0] if k_quantizer else None
        v_quantizer = v_quantizer.shards[0] if v_quantizer else None

        k_shard, v_shard = self.kv_caches[pipeline].read(
            state=state,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
            k_quantizer=k_quantizer,
            v_quantizer=v_quantizer,
        )

        # Don't have to transfer since state is already on the correct device
        devices = self.config.devices_for_pipeline(pipeline)
        key = ReplicatedTensor(ts=[k_shard], devices=devices)
        value = ReplicatedTensor(ts=[v_shard], devices=devices)
        return key, value

    def write(
        self,
        state: CacheAllocation,
        *,
        cache_partitions: List[ReplicatedTensor],
        transformer_block_index: int,
        page_ids: ReplicatedTensor,
        start_positions: ReplicatedTensor | None,
    ) -> None:
        pipeline = self.config.pipeline_for_block(transformer_block_index)
        transformer_block_index = self.adjust_index(transformer_block_index)

        assert len(page_ids.shards) == 1, "Tensor parallelism not supported."
        assert all(
            len(cp.shards) == 1 for cp in cache_partitions
        ), "Tensor parallelism not supported."
        if start_positions is not None:
            assert len(start_positions.shards) == 1, "Tensor parallelism not supported."

        state = CacheAllocation([state[pipeline]])
        cache_partitions = [cp.shards[0] for cp in cache_partitions]
        page_ids = page_ids.shards[0]
        start_positions = start_positions.shards[0] if start_positions else None

        self.kv_caches[pipeline].write(
            state=state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            page_ids=page_ids,
            start_positions=start_positions,
        )

    def write_timestep(
        self,
        state: CacheAllocation,
        *,
        cache_partitions: List[ReplicatedTensor],
        transformer_block_index: int,
        seq_positions: ReplicatedTensor,
        page_ids: ReplicatedTensor,
    ) -> None:
        pipeline = self.config.pipeline_for_block(transformer_block_index)
        transformer_block_index = self.adjust_index(transformer_block_index)

        assert all(
            len(cp.shards) == 1 for cp in cache_partitions
        ), "Tensor parallelism not supported."
        assert len(seq_positions.shards) == 1, "Tensor parallelism not supported."
        assert len(page_ids.shards) == 1, "Tensor parallelism not supported."

        state = CacheAllocation([state[pipeline]])
        cache_partitions = [cp.shards[0] for cp in cache_partitions]
        seq_positions = seq_positions.shards[0]
        page_ids = page_ids.shards[0]

        self.kv_caches[pipeline].write_timestep(
            state=state,
            cache_partitions=cache_partitions,
            transformer_block_index=transformer_block_index,
            seq_positions=seq_positions,
            page_ids=page_ids,
        )


def build_cache(
    transformer_block_count: int,
    attn_head_count: int,
    attn_head_dim: int,
    cache_partition_count: int = 2,
    block_seq_stride: int = 16,
    cache_dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    parallelism_config: ParallelismConfig | None = None,
) -> PagedKVCache:
    kwargs = dict(
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        cache_partition_count=cache_partition_count,
        block_seq_stride=block_seq_stride,
        cache_dtype=cache_dtype,
        device=device,
    )

    if parallelism_config is None or parallelism_config.pipeline_size == 1:
        PagedKVCacheClazz = DefaultPagedKVCache
        kwargs["transformer_block_count"] = transformer_block_count
    else:
        PagedKVCacheClazz = PipelinedPagedKVCache
        kwargs["parallelism_config"] = parallelism_config

    return PagedKVCacheClazz(**kwargs)


def build_cache_from_config(config: LlamaModelConfig) -> PagedKVCache:
    return build_cache(
        transformer_block_count=config.hp.block_count,
        attn_head_count=config.hp.attention_head_count_kv,
        attn_head_dim=config.hp.attn_head_dim,
        block_seq_stride=config.block_seq_stride,
        cache_dtype=config.kv_cache_dtype or config.attention_dtype,
        device=config.device,
        parallelism_config=config.parallelism_config,
    )


class PagedAttention(ABC):
    """abstract class for paged attention interface"""

    @abstractmethod
    def allocate(self, page_count: int) -> CacheAllocation:
        ...

    @abstractmethod
    def read(
        self,
        state: CacheAllocation,
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor | ReplicatedTensor | None = None,
    ):
        ...

    @abstractmethod
    def write_timestep(
        self,
        state: CacheAllocation,
        cache_partitions: list[torch.Tensor | QuantizedTensor],
        *,
        transformer_block_index: int,
        seq_positions: torch.Tensor | ReplicatedTensor,
        page_ids: torch.Tensor | ReplicatedTensor,
    ):
        ...

    @abstractmethod
    def write(
        self,
        state: CacheAllocation,
        cache_partitions: list[torch.Tensor | QuantizedTensor],
        *,
        transformer_block_index: int,
        page_ids: torch.Tensor | ReplicatedTensor,
        start_positions: torch.Tensor | ReplicatedTensor | None = None,
    ):
        ...

    @abstractmethod
    def attention(
        self,
        *,
        q: torch.Tensor | ReplicatedTensor,
        k: torch.Tensor | ReplicatedTensor,
        v: torch.Tensor | ReplicatedTensor,
        head_count_attn: int,
        cache_quantizer: QuantizerTensor | ReplicatedTensor | None,
        attention_kernel: str,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor | ReplicatedTensor] = None,
        mask: Optional[torch.Tensor | ReplicatedTensor] = None,
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor | ReplicatedTensor] = None,
    ) -> torch.Tensor | ReplicatedTensor:
        ...

    @abstractmethod
    def forward_decode(
        self,
        *,
        q: torch.Tensor | ReplicatedTensor,
        k: torch.Tensor | ReplicatedTensor,
        v: torch.Tensor | ReplicatedTensor,
        cache_state: CacheAllocation,
        seq_block_ids: torch.Tensor | ReplicatedTensor,
        start_positions: torch.Tensor | ReplicatedTensor,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        seq_lens: torch.Tensor | None,
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | ReplicatedTensor:
        ...

    @abstractmethod
    def forward_prefill(
        self,
        *,
        q: torch.Tensor | ReplicatedTensor,
        k: torch.Tensor | ReplicatedTensor,
        v: torch.Tensor | ReplicatedTensor,
        cache_state: CacheAllocation,
        seq_block_ids: torch.Tensor | ReplicatedTensor,
        start_positions: Optional[torch.Tensor] = None,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: QuantizerTensor | ReplicatedTensor | None,
        fake_quant: Optional[bool],
        seq_lens: torch.Tensor | None,
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | ReplicatedTensor:
        ...

    @abstractmethod
    def paged_attention(
        self,
        *,
        q: torch.Tensor | ReplicatedTensor,
        k: torch.Tensor | ReplicatedTensor,
        v: torch.Tensor | ReplicatedTensor,
        cache_state: CacheAllocation,
        seq_lens: torch.Tensor | ReplicatedTensor | None,
        seq_block_ids: torch.Tensor | ReplicatedTensor,
        start_positions: torch.Tensor | ReplicatedTensor | None,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: QuantizerTensor | ReplicatedTensor | None,
        fake_quant: Optional[bool],
        softcap: Optional[float],
        scale: Optional[float],
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | ReplicatedTensor:
        ...


class PagedMHAttention(PagedAttention):
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
        transformer_block_index: int,
        attn_dtype: torch.dtype = torch.float32,
        activation_dtype: torch.dtype = torch.float32,
        use_rope: bool,
        attention_chunk_size: int | None,
        kv_cache: KVCache,
        k_quantizer: StaticScaledQuantizer | None = None,
        v_quantizer: StaticScaledQuantizer | None = None,
    ):
        self.transformer_block_index = transformer_block_index
        self.block_seq_stride = kv_cache.block_seq_stride
        self.attn_dtype = attn_dtype
        self.kv_cache = kv_cache
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer
        self.activation_dtype = activation_dtype
        self.attention_chunk_size = attention_chunk_size
        self.use_rope = use_rope

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
            k_quantizer=self.k_quantizer,
            v_quantizer=self.v_quantizer,
        )

    def write_timestep(
        self,
        state: CacheAllocation,
        cache_partitions: List[torch.Tensor | QuantizedTensor],
        *,
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
        cache_partitions: List[torch.Tensor | QuantizedTensor],
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

    def attention(
        self,
        *,
        q: torch.Tensor | ReplicatedTensor,
        k: torch.Tensor | ReplicatedTensor,
        v: torch.Tensor | ReplicatedTensor,
        head_count_attn: int,
        cache_quantizer: QuantizerTensor | ReplicatedTensor | None,
        attention_kernel: str,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor | ReplicatedTensor] = None,
        mask: Optional[torch.Tensor | ReplicatedTensor] = None,
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor | ReplicatedTensor] = None,
        cast_kv: bool = True,
    ) -> torch.Tensor | ReplicatedTensor:
        # Fake quant is already dequantized when stored in the cache.

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if cast_kv:
            # GQA Attention has a mechanism that enables a performance improvement but
            # requires moving the cache dequantization to earlier
            if cache_quantizer and not fake_quant:
                k_planes = {"qs": k}
                k = ops.dequantize(
                    k_planes, quantizer=cache_quantizer, dtype=self.attn_dtype
                )
                v_planes = {"qs": v}
                v = ops.dequantize(
                    v_planes, quantizer=cache_quantizer, dtype=self.attn_dtype
                )

        return ops.scaled_dot_product_attention(
            q=q,  # [bs, ..., sl, dim]
            k=k,  # [bs, ..., sl, dim]
            v=v,  # [bs, ..., sl, dim]
            a=mask,  # [bs, ..., sl, sl] or None
            is_causal=mask is None,  # assumes causal masking when true
            scale=scale,  # defaults to 1/sqrt(dim)
            softcap=softcap,
            impl=attention_kernel,  # if none, automatically select a kernel
            sink=sink,
            sliding_window=sliding_window,
        )

    def forward_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: CacheAllocation,
        seq_block_ids: torch.Tensor,
        start_positions: torch.Tensor,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        seq_lens: torch.Tensor | None,
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
    ):
        # Write our one updated cache row into the cache.
        self.write_timestep(
            cache_state,
            cache_partitions=[k, v],
            transformer_block_index=self.transformer_block_index,
            seq_positions=start_positions,
            page_ids=seq_block_ids,
        )

        return self.paged_attention(
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            attention_kernel=attention_kernel,
            head_count_attn=head_count_attn,
            cache_quantizer=cache_quantizer,
            start_positions=start_positions,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            sliding_window=sliding_window,
            sink=sink,
        )

    def forward_prefill(
        self,
        *,
        q: torch.Tensor | ReplicatedTensor,
        k: torch.Tensor | ReplicatedTensor,
        v: torch.Tensor | ReplicatedTensor,
        cache_state: CacheAllocation,
        seq_block_ids: torch.Tensor | ReplicatedTensor,
        start_positions: Optional[torch.Tensor] = None,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: QuantizerTensor | ReplicatedTensor | None,
        fake_quant: Optional[bool],
        seq_lens: torch.Tensor | None,
        softcap: Optional[float] = None,
        scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | ReplicatedTensor:
        self.write(
            cache_state,
            cache_partitions=[k, v],
            transformer_block_index=self.transformer_block_index,
            page_ids=seq_block_ids,
            start_positions=start_positions,
        )

        return self.paged_attention(
            q=q,
            k=k,
            v=v,
            cache_state=cache_state,
            seq_lens=seq_lens,
            seq_block_ids=seq_block_ids,
            start_positions=start_positions,
            attention_kernel=attention_kernel,
            head_count_attn=head_count_attn,
            cache_quantizer=cache_quantizer,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            sliding_window=sliding_window,
            sink=sink,
        )

    def paged_attention(
        self,
        *,
        q: torch.Tensor,
        k,
        v,
        cache_state: CacheAllocation,
        seq_lens: torch.Tensor | None,
        seq_block_ids: torch.Tensor,
        start_positions: torch.torch.Tensor | None,
        attention_kernel: str,
        head_count_attn: int,
        cache_quantizer: Optional[QuantizerTensor],
        fake_quant: Optional[bool],
        softcap: Optional[float],
        scale: Optional[float],
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
    ):
        # Restore from the cache.
        if start_positions is not None:
            k, v = self.read(
                cache_state,
                transformer_block_index=self.transformer_block_index,
                page_ids=seq_block_ids,
            )

        is_prefill = q.shape[1] != 1
        if is_prefill:
            source_len = seq_block_ids.shape[1] * self.block_seq_stride
            target_len = q.shape[1]
            input_mask = create_input_mask(seq_lens, source_len)
            mask = create_attention_mask(
                input_mask,
                start_positions,
                source_len=source_len,
                target_len=target_len,
                attention_dtype=self.activation_dtype,
            )
            use_chunked_attention_mask = self.attention_chunk_size is not None
            if use_chunked_attention_mask and self.use_rope:
                mask = create_chunked_attention_mask(mask, self.attention_chunk_size)
        else:
            input_mask = create_input_mask(
                seq_lens,
                seq_block_ids.shape[1] * self.block_seq_stride,
            )
            mask = create_attention_mask_for_decode(
                input_mask, attention_dtype=self.activation_dtype
            )
            if self.attention_chunk_size is not None:
                raise NotImplementedError("Chunked attention not supported in decode.")

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
            sliding_window=sliding_window,
            sink=sink,
        )


class PagedGQAttention(PagedMHAttention):
    def attention(
        self,
        *,
        q: torch.Tensor | ReplicatedTensor,
        k: torch.Tensor | ReplicatedTensor,
        v: torch.Tensor | ReplicatedTensor,
        head_count_attn: int,
        cache_quantizer: QuantizerTensor | ReplicatedTensor | None,
        attention_kernel: str,
        fake_quant: Optional[bool],
        softcap: Optional[float] = None,
        scale: Optional[torch.Tensor | ReplicatedTensor] = None,
        mask: Optional[torch.Tensor | ReplicatedTensor] = None,
        sliding_window: Optional[int] = None,
        sink: Optional[torch.Tensor | ReplicatedTensor] = None,
    ) -> torch.Tensor | ReplicatedTensor:
        gqa_n_rep = head_count_attn // self.kv_cache.attn_head_count
        assert gqa_n_rep > 0
        if gqa_n_rep > 1:
            bs, slen, n_kv_heads, head_dim = k.shape
            if cache_quantizer and not fake_quant:
                k_planes = {"qs": k}
                k = ops.dequantize(
                    k_planes, quantizer=cache_quantizer, dtype=self.attn_dtype
                )
                v_planes = {"qs": v}
                v = ops.dequantize(
                    v_planes, quantizer=cache_quantizer, dtype=self.attn_dtype
                )
            k = ops.expand(
                k.unsqueeze(-2), (bs, slen, n_kv_heads, gqa_n_rep, head_dim)
            ).flatten(2, 3)
            bs, slen, n_kv_heads, head_dim = v.shape
            v = ops.expand(
                v.unsqueeze(-2), (bs, slen, n_kv_heads, gqa_n_rep, head_dim)
            ).flatten(2, 3)

        return super().attention(
            q=q,
            k=k,
            v=v,
            head_count_attn=head_count_attn,
            cache_quantizer=cache_quantizer,
            attention_kernel=attention_kernel,
            fake_quant=fake_quant,
            softcap=softcap,
            scale=scale,
            mask=mask,
            sliding_window=sliding_window,
            sink=sink,
            cast_kv=False,
        )


class PagedMLAttention(PagedMHAttention):
    """
    This subclass is intentionally left empty. All behavioral differences between
    PagedMLAttention and PagedMHAttention are currently handled by PagedLlamaAttentionBlock.
    Do not remove this class, as it serves as a placeholder for future extensions
    and maintains compatibility with the attention module's interface.
    """

    pass
