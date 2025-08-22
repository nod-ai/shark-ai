# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Union, Sequence
import math
import itertools
import torch
from torch import Tensor
from sharktank.types import (
    AnyTensor,
    PrimitiveTensor,
    unbox_tensor,
    SplitPrimitiveTensor,
    ReplicatedTensor,
    QuantizedTensor,
    TensorScaledLayout,
    BlockScaledLayout,
)
from sharktank.ops._registry import overridable
from sharktank.ops.sharding.utils import _reshape_infer_dynamic_dim


@overridable(dispatch_args=(0,))
def view(
    tensor: AnyTensor, shape: List[int] | None = None, dtype: torch.dtype | None = None
) -> AnyTensor:
    """See torch.Tensor.view"""
    ...


@view.override(Tensor)
def view_default(
    tensor: Union[Tensor, PrimitiveTensor],
    shape: List[int] | None,
    dtype: torch.dtype | None,
) -> Tensor:
    tensor = unbox_tensor(tensor)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if shape is not None:
        tensor = tensor.view(shape)
    return tensor


@view.override(QuantizedTensor)
def view_QuantizedTensor(tensor: QuantizedTensor, shape):
    unpacked = tensor.unpack()
    if isinstance(unpacked, TensorScaledLayout):
        new_qs = unpacked._qs.view(shape)
        layout = TensorScaledLayout(
            shape=shape,
            d=unpacked.d,
            qs=new_qs,
            m=unpacked.m,
            metadata=unpacked.metadata,
        )
        return QuantizedTensor(shape=shape, layout=layout)
    if isinstance(unpacked, BlockScaledLayout):
        return view_block_scaled(tensor, shape)
    raise NotImplementedError(f"QuantizedTensor.view with {type(unpacked)}")


def view_block_scaled(tensor, shape):
    unpacked = tensor.unpack()
    old_shape = tensor.shape
    d = unpacked.d.view(-1)
    m = unpacked.m.view(-1) if unpacked.m is not None else None
    qs = unpacked.qs.view(-1)

    block_size = unpacked.block_size()
    n_blocks_old = math.prod(old_shape) // block_size
    n_blocks_new = math.prod(shape) // block_size
    assert n_blocks_old == n_blocks_new

    new_d_shape = [n_blocks_new, unpacked.d.shape[-1]]
    new_m_shape = [n_blocks_new, unpacked.m.shape[-1]] if m is not None else None
    new_qs_shape = [n_blocks_new, unpacked.qs.shape[-1]]

    d = d.view(new_d_shape)
    m = m.view(new_m_shape) if m is not None else None
    qs = qs.view(new_qs_shape)

    layout = unpacked.create(
        shape=shape,
        d=d,
        m=m,
        qs=qs,
        metadata=unpacked.metadata,
    )
    return QuantizedTensor(shape=shape, layout=layout)


@view.override(ReplicatedTensor)
def view_replicated(
    tensor: ReplicatedTensor, shape: List[int] | None, dtype: torch.dtype | None
) -> ReplicatedTensor:
    return ReplicatedTensor(ts=[view(shard, shape, dtype) for shard in tensor.shards])


@view.override(SplitPrimitiveTensor)
def view_split(
    tensor: SplitPrimitiveTensor, shape: List[int] | None, dtype: torch.dtype | None
) -> SplitPrimitiveTensor:
    assert dtype is None, "Not supported"
    shard_dim = tensor.shard_dim
    mapping = _calculate_view_dimension_mapping(from_shape=tensor.shape, to_shape=shape)
    if len(mapping[shard_dim]) != 1:
        if tensor.shape[tensor.shard_dim] % tensor.shard_count != 0:
            raise ValueError(
                "Only splitting a dimension that is multiple of the shard count is supported"
            )
        if shape[tensor.shard_dim] % tensor.shard_count != 0:
            raise ValueError(
                "The resulting leading splitting dimension must be multiple of the shard count"
            )

    # Account for collapsed or expanded dims
    collapsed_dims = []
    delta = 0
    for from_dim, to_dims in enumerate(mapping[: shard_dim + 1]):
        if len(to_dims) > 1:
            # Expanded dims move shard_dim to the right by 1 for each new dim.
            if from_dim == shard_dim:
                pass  # Do nothing since we want to shard based on the leading dim if the shard_dim is expanded.
            else:
                delta += len(to_dims) - 1
        # A to_dim can be split to be both expand itself and be collapsed with others, must check.
        for to_dim in to_dims:
            # Collapsed dims move shard_dim to the left by 1 for each dim after the first.
            if to_dim in collapsed_dims:
                delta -= 1
            collapsed_dims.append(to_dim)
    # Account for extra dims of size 1
    dims_not_seen = [i for i in range(min(mapping[shard_dim]))]
    for to_dims in mapping[:shard_dim]:
        for to_dim in to_dims:
            if to_dim in dims_not_seen:
                dims_not_seen.remove(to_dim)

    shard_dim += delta + len(dims_not_seen)

    new_shard_shape = list(shape)
    # NOTE: dynamic shard_dim is handled implicitly because of int division.
    new_shard_shape[shard_dim] //= tensor.shard_count
    shards = [view(shard, new_shard_shape) for shard in tensor.shards]
    res = SplitPrimitiveTensor(shard_dim=shard_dim, ts=shards)
    assert math.prod(res.shape) == math.prod(tensor.shape)
    return res


def _calculate_view_dimension_mapping(
    from_shape: Sequence[int], to_shape: Sequence[int]
) -> List[List[int]]:
    """
    Calculate a mapping from the dimensions in `from_shape` to those in `to_shape`.
    """
    from_shape, to_shape = list(from_shape), list(to_shape)
    assert len(from_shape) > 0 and len(to_shape) > 0, "Scalars not supported"
    assert all(d != 0 for d in from_shape + to_shape), "Zero dimensions not supported"
    from_shape, to_shape = _reshape_infer_dynamic_dim(list(from_shape), list(to_shape))

    # Trivial cases
    if len(from_shape) == 1:
        return [[i for i in range(len(to_shape))]]
    if len(to_shape) == 1:
        return [[0] for _ in range(len(from_shape))]

    def _get_cumulative_boundaries(shape: Sequence[int]) -> List[int]:
        """
        Get the cumulitive number of elements at the start of each dimension.
        Add an extra 1 at the start to represent the start of the first dimension.
        For example, for shape (2, 3, 4) it returns [1, 2, 6, 24].
        """
        return [1] + list(itertools.accumulate(shape, lambda x, y: x * y))

    bounds_to = _get_cumulative_boundaries(to_shape)
    bounds_from = _get_cumulative_boundaries(from_shape)

    mapping = [[] for _ in range(len(from_shape))]
    to_dim_idx_start = 0
    for from_dim in range(len(from_shape)):
        from_bound_start = bounds_from[from_dim]
        from_bound_end = bounds_from[from_dim + 1]

        to_dim = to_dim_idx_start
        while to_dim < len(to_shape):
            to_bound_start = bounds_to[to_dim]
            to_bound_end = bounds_to[to_dim + 1]

            # Check if the two ranges overlap
            overlap_start = max(to_bound_start, from_bound_start)
            overlap_end = min(to_bound_end, from_bound_end)
            range_overlaps = overlap_start < overlap_end

            # Special case for dim of size 1
            size_one_dim_overlap = False
            if from_bound_start == from_bound_end:  # `from_dim` is 1
                if (
                    from_bound_start >= to_bound_start
                    and from_bound_start < to_bound_end
                ):
                    # `from_dim` is within the range of `to_dim`.
                    # E.g. [5, 1, 6] to [5, 6]
                    size_one_dim_overlap = True
                elif (
                    from_bound_start == to_bound_start
                    and from_bound_end == to_bound_start
                ):
                    size_one_dim_overlap = True

            if range_overlaps or size_one_dim_overlap:
                # Overlap exists
                assert to_dim not in mapping[from_dim]
                mapping[from_dim].append(to_dim)

                if to_bound_end >= from_bound_end:
                    # We have exhausted the current `from_dim`
                    if to_bound_end == from_bound_end:
                        # This `to_dim` ends *exactly* at the end of the current `from_dim`.
                        # This `to_dim` is exhausted, start next search with next `to_dim`.
                        to_dim_idx_start = to_dim + 1
                    else:  # to_bound_end > from_bound_end
                        # This `to_dim` ends *after* the current `from_dim` ends.
                        # We need to check the next `from_dim` for the current `to_dim`;
                        # This `to_dim` is split across multiple `from_dim`s.
                        to_dim_idx_start = to_dim
                    # Found all contributions of this `from_dim`, more to the next.
                    break
                else:  # to_bounds_end < from_bounds_end
                    # This to_dim ends *before* the current `from_dim` ends.
                    # We need to check the next to_dim for the current `from_dim`.
                    to_dim += 1
            elif to_bound_start > from_bound_end:
                # This `to_dim` starts *after* the current `from_dim` ends.
                # No further `to_dim`s will overlap this `from_dim`.
                # The next search should start from this `to_dim`.
                to_dim_idx_start = to_dim
                break
            else:  # to_bounds_end <= from_bounds_start
                # This `to_dim` ends *before* or *at* the start of the current `from_dim`.
                # Move to check the next `to_dim` for the current `from_dim`.
                to_dim += 1
        # Update search start if inner loop finishes by exhaustion
        if to_dim == len(to_shape):
            to_dim_idx_start = to_dim

        # Handle empty mapping for size 1 dimensions that didn't get mapped (happens if this is trailing 1)
        if from_shape[from_dim] == 1 and not mapping[from_dim]:
            last_valid_idx = len(to_shape) - 1
            mapping[from_dim].append(last_valid_idx)

    return mapping
