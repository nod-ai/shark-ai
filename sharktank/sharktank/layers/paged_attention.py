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

__all__ = ["PagedAttention"]


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
        block_to_device_lookup: tuple[tuple[int, ...], ...] | None = None,
    ):
        self.transformer_block_count = transformer_block_count
        self.head_count_kv = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride
        self.shard_count = shard_count

        if block_to_device_lookup is None:
            block_to_device_lookup = tuple(
                tuple(range(self.shard_count))
                for _ in range(self.transformer_block_count)
            )
        assert len(block_to_device_lookup) == transformer_block_count
        block_to_pipeline_lookup = [0]
        pipeline_to_device_lookup = [block_to_device_lookup[0]]
        pipeline = 0
        for block in range(1, transformer_block_count):
            ds_prev, ds_curr = (
                block_to_device_lookup[block - 1],
                block_to_device_lookup[block],
            )
            assert all(d for d in ds_prev) >= 0
            assert all(d for d in ds_curr) >= 0
            if not all(d_prev == d_curr for d_prev, d_curr in zip(ds_prev, ds_curr)):
                pipeline += 1
                pipeline_to_device_lookup.append(ds_curr)
            block_to_pipeline_lookup.append(pipeline)

        self.pipeline_to_device_lookup = tuple(pipeline_to_device_lookup)
        self.block_to_pipeline_lookup = tuple(block_to_pipeline_lookup)
        self.pipeline_count = len(pipeline_to_device_lookup)

        if attn_head_count % shard_count != 0:
            raise ValueError(
                f"The attention head count {attn_head_count} must be a multiple of the tensor parallelism size {shard_count}."
            )

        self.pipeline_to_block_count = tuple(
            sum(1 for block in block_to_pipeline_lookup if block == i)
            for i in range(self.pipeline_count)
        )
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

        if self.shard_count == 1 and self.pipeline_count == 1:
            assert all(
                not isinstance(page_slab, SplitPrimitiveTensor) for page_slab in state
            )
            return [
                page_slab.unflatten(1, self.sub_page_dims[pipeline])
                for pipeline, page_slab in enumerate(state)
            ]

        assert all(page_slab.shard_count == self.shard_count for page_slab in state)
        unflattened = []
        for pipeline, page_slab in enumerate(state):
            shards = [
                shard.unflatten(1, self.sub_page_dims[pipeline])
                for shard in page_slab.shards
            ]
            unflattened.append(
                (
                    SplitPrimitiveTensor(
                        ts=shards, shard_dim=4, devices=page_slab.devices
                    )
                    if len(shards) > 1
                    else ReplicatedTensor(ts=shards, devices=page_slab.devices)
                )
            )
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
            # TODO: Do I need to make copies here, or are views enough?
            assert (
                self.pipeline_to_block_count[pipeline] != 1
            ), "1 tensor per pipeline not supported. dim gets collapsed"
            i_min = sum(self.pipeline_to_block_count[:pipeline])
            i_max = i_min + self.pipeline_to_block_count[pipeline]
            if self.shard_count == 1:
                sharded_page_table = ops.replicate(
                    page_table[:, i_min:i_max, ...],
                    count=1,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
            else:
                sharded_page_table = ops.reshard_split(
                    page_table[:, i_min:i_max, ...],
                    dim=4,
                    count=self.shard_count,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
            shards_flattened = [
                ops.flatten(shard, start_dim=1) for shard in sharded_page_table.shards
            ]
            flat_sharded_page_tables.append(
                SplitPrimitiveTensor(
                    ts=shards_flattened,
                    shard_dim=1,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
                if self.shard_count > 1
                else ReplicatedTensor(
                    ts=shards_flattened,
                    devices=self.pipeline_to_device_lookup[pipeline],
                )
            )
        return flat_sharded_page_tables

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
                    [page_count, self.page_slab_flat_dims[pipeline]],
                    dtype=self.cache_dtype,
                    device=self.device,
                )
                for _ in range(self.shard_count)
            ]
            for pipeline in range(self.pipeline_count)
        ]

        if self.shard_count == 1 and self.pipeline_count == 1:
            return shards[0]

        return [
            (
                SplitPrimitiveTensor(ts=shards[i], shard_dim=1, devices=devices)
                if len(shards[i]) > 1
                else ReplicatedTensor(ts=shards[i], devices=devices)
            )
            for i, devices in enumerate(self.pipeline_to_device_lookup)
        ]

    def read(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        seq_len: int,
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
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]

        bs, block_seq_len, *_ = page_ids.shape
        # Blocks dim 1,2 according to the configured block stride.
        blocked_shape = [
            bs,
            block_seq_len,
            self.cache_partition_count,
            self.block_seq_stride,
            self.head_count_kv // self.shard_count,
            self.attn_head_dim,
        ]

        # Gather both partitions and split post gather. This is more
        # computationally efficient without gather fusion:
        subblock_table = page_table.flatten(start_dim=0, end_dim=1)
        page_stride = self.pipeline_to_block_count[
            self.block_to_pipeline_lookup[transformer_block_index]
        ]

        transformer_block_index = torch.full(
            (bs, block_seq_len), transformer_block_index, device=self.device
        )
        subblock_ids = page_ids * page_stride + transformer_block_index
        selected = ops.index_select(subblock_table, 0, subblock_ids.flatten(0, 1))

        selected = selected.unflatten(0, blocked_shape[:2])
        key = selected[:, :, 0, :seq_len].flatten(1, 2)[:, :seq_len]
        value = selected[:, :, 1, :seq_len].flatten(1, 2)[:, :seq_len]

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
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]
        page_table = page_table.flatten(0, 3)
        bs, *_ = seq_positions.shape
        assert len(cache_partitions) == self.cache_partition_count

        # [bs, 1, atten_head_count, attn_head_dim]
        for idx, cache_partition in enumerate(cache_partitions):
            # [bs, 1]
            page_index = seq_positions // self.block_seq_stride

            page_id = ops.gather(page_ids, dim=1, index=page_index.unsqueeze(1))
            page_offset = (seq_positions % self.block_seq_stride).unsqueeze(1)

            # [1, 1]
            if isinstance(seq_positions, ReplicatedTensor):
                partitions = [
                    torch.tensor(idx, device=device).unsqueeze(0)
                    for _ in range(seq_positions.shard_count)
                ]

                transformer_block = [
                    torch.full((bs, 1), transformer_block_index, device=device)
                    for _ in range(seq_positions.shard_count)
                ]

                devices = self.pipeline_to_device_lookup[
                    self.block_to_pipeline_lookup[transformer_block_index]
                ]
                partitions = ReplicatedTensor(ts=partitions, devices=devices)
                transformer_block = ReplicatedTensor(
                    ts=transformer_block, devices=devices
                )
            else:
                partitions = torch.tensor(idx, device=device).unsqueeze(0)
                transformer_block = torch.full(
                    (bs, 1), transformer_block_index, device=device
                )

            partitions = partitions.repeat(bs, 1)

            transformer_block_count_in_pipeline = self.pipeline_to_block_count[
                self.block_to_pipeline_lookup[transformer_block_index]
            ]

            index = page_id
            index = index * transformer_block_count_in_pipeline + transformer_block
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
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]
        bs, block_seq_len, *_ = page_ids.shape

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        transformer_block_count_in_pipeline = self.pipeline_to_block_count[
            self.block_to_pipeline_lookup[transformer_block_index]
        ]
        page_stride = transformer_block_count_in_pipeline * self.cache_partition_count
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

            # mask = mask != float("-inf")
            output = ops.scaled_dot_product_attention(
                q=q,  # [bs, ..., sl, dim]
                k=k,  # [bs, ..., sl, dim]
                v=v,  # [bs, ..., sl, dim]
                a=mask,  # [bs, ..., sl, sl]
                is_causal=mask is None,  # assumes causal masking when true
                scale=None,  # defaults to 1/sqrt(dim)
            )

            #####################################################
            # Debug tracing
            # TODO: delete this when done
            import safetensors.torch

            print(f"Mask dtype = {mask.dtype}")
            print(f"values = {mask.unique()}")
            tensor_dict = {
                "q": q.contiguous(),
                "k": k.contiguous(),
                "v": v.contiguous(),
                "mask": mask,
                "output": output.contiguous(),
            }
            import os

            attention_idx = int(os.environ.get("sharktank_attention_idx", default="0"))
            safetensors.torch.save_file(
                tensors=tensor_dict,
                filename=f"/home/bpetkant/ws/sharktank/experiments/llama4/attention_trace{attention_idx}.safetensors",
            )
            os.environ["sharktank_attention_idx"] = str(attention_idx + 1)
            return output
            #####################################################

    def forward_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_state: list[torch.Tensor],
        seq_block_ids: torch.Tensor,
        kv_seq_len: int,
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
            seq_len=kv_seq_len,
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
