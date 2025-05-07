# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

from typing import Callable, Sequence, Any
from sharktank.utils import tree


def call_trivially_replicable(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Call an function with replicated tensor arguments that can be called with
    unsharded tensors.

    It is expected that all replicated tensors have the same number of shards.
    The function will be called with each set of matching shard indices and
    replicated tensor(s) will be constructed from the resulting shards.

    This wrapper handles tee-like structures in the arguments and results. It will
    traverse them and handle nested tensors. Note that sequence and dictionary types
    will be reconstructed with dict and tuple types, so the underlying function fn
    must be polymorphic in this regard and not expect concrete types.

    When collecting the results all non-tensor results are taken from the first device.
    """

    from sharktank.types import (
        ReplicatedTensor,
        is_any_tensor,
    )

    flat_args = tree.flatten(
        (
            args,
            kwargs,
        ),
        is_leaf=_is_leaf,
    )
    first_replicated_tensor_arg = [
        arg for arg in flat_args if isinstance(arg, ReplicatedTensor)
    ]
    first_replicated_tensor_arg = (
        None
        if len(first_replicated_tensor_arg) == 0
        else first_replicated_tensor_arg[0]
    )
    if first_replicated_tensor_arg is None:
        # No replicated tensor arguments, just call the function.
        return fn(*args, **kwargs)

    shard_count = first_replicated_tensor_arg.shard_count

    per_shard_args_and_kwargs = [
        _extract_per_shard_args_and_kwargs(i, args, kwargs) for i in range(shard_count)
    ]
    per_shard_results = [
        fn(*per_shard_args, **per_shard_kwargs)
        for per_shard_args, per_shard_kwargs in per_shard_args_and_kwargs
    ]

    def reduce_to_list_of_tensors_fn(a: Any, b: Any) -> Any:
        if not is_any_tensor(b):
            return a
        a.append(b)
        return a

    def tensor_to_empty_list_if_tensor(x: Any) -> Any:
        return [] if is_any_tensor(x) else x

    reduce_initial = tree.map_leaves(
        per_shard_results[0], f=tensor_to_empty_list_if_tensor, is_leaf=_is_leaf
    )

    # Make a lists of tensor shards that correspond to the same replicated tensor.
    reduced_results = tree.reduce_horizontal(
        fn=reduce_to_list_of_tensors_fn,
        trees=per_shard_results,
        is_leaf=_is_leaf,
        initial=reduce_initial,
    )

    def make_replicated_tensor_if_tensor_collection(x: Any) -> Any:
        if _is_tensor_collection(x):
            return ReplicatedTensor(ts=x)
        return x

    result_with_replicated_tensor = tree.map_leaves(
        reduced_results,
        f=make_replicated_tensor_if_tensor_collection,
        is_leaf=_is_leaf_or_tensor_collection,
    )
    return result_with_replicated_tensor


def trivially_replicable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator to turn an trivially replicable operation into a replicated
    operation.

    An operation is trivially replicable if the replicated variant just executes the
    operation on all devices with the respective shards.

    E.g.
    ```
    @trivially_replicable
    def fn(a: torch.Tensor) -> torch.Tensor:
        return a

    arg = torch.Tensor([1, 2, 3])
    shard_count = 2
    replicated_arg = ReplicatedTensor(
        ts=arg, shard_count=shard_count
    )
    fn(replicated_arg)
    ```

    ```
    ReplicatedTensor(<unnamed>, [3], shard_count=2 of [3])
    ```
    """

    def wrapper(*args, **kwargs):
        return call_trivially_replicable(fn, args, kwargs)

    return wrapper


def _extract_per_shard_args_and_kwargs(
    shard_index: int, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return tree.map_leaves(
        (args, kwargs),
        f=functools.partial(
            _extract_tensor_shard_if_replicate, shard_index=shard_index
        ),
        is_leaf=_is_leaf,
    )


def _extract_tensor_shard_if_replicate(v: Any, shard_index: int) -> Any:
    from sharktank.types import (
        ReplicatedTensor,
    )

    if isinstance(v, ReplicatedTensor):
        return v.shards[shard_index]
    return v


def _is_leaf(v: Any) -> bool:
    from sharktank.types import is_any_tensor

    return is_any_tensor(v) or tree.is_leaf_default(v)


def _is_leaf_or_tensor_collection(x: Any) -> bool:
    return _is_leaf(x) or _is_tensor_collection(x)


def _is_tensor_collection(x: Any) -> bool:
    from sharktank.types import is_any_tensor

    return isinstance(x, Sequence) and all(is_any_tensor(v) for v in x)
