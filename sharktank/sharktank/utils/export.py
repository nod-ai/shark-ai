# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Optional, Any
from os import PathLike
import functools

import torch
from torch.utils._pytree import PyTree, _is_leaf

import iree.turbine.aot as aot
from iree.turbine.aot import DeviceAffinity, FxProgramsBuilder
from torch.utils._pytree import tree_structure, tree_unflatten, tree_flatten
from amdsharktank.types.tensors import ShardedTensor
from amdsharktank.types.theta import mark_export_external_theta
from amdsharktank.layers import BaseLayer, ThetaLayer


def flatten_signature(
    *sample_args: list[PyTree], **sample_kwargs: dict[str, PyTree]
) -> Callable[[Callable], Any]:
    """Decorator that flattens the signature of a function using PyTorch's type
    registration.
    It will flatten the same way torch PyTorch does, returning a function that accepts
    and returns a flat list of torch.Tensor.
    The decorator requires sample arguments of the unflattened function.

    ```
    @flatten_signature(
        {
            "a1": SplitPrimitiveTensor(ts=[torch.tensor([1])], shard_dim=0),
            "a2": torch.tensor([2]),
        },
        [DefaultPrimitiveTensor(data=torch.tensor([3]))]
    )
    def f(a, b):
        return a["a1"], b
    ```

    will result in a function with signature

    ```
    (
        torch.Tensor of size 1,
        torch.Tensor of size 2,
        torch.Tensor of size 3,
    ) -> (
        torch.Tensor of size 1,
        torch.Tensor of size 2,
    )
    ```
    """
    flat_sample_args, args_tree_spec = tree_flatten(sample_args)
    n_args = len(flat_sample_args)
    kwargs_tree_spec = tree_structure(sample_kwargs)

    def _decorator(f: Callable) -> Callable:
        def _wrapper(*flat_args: list[Any]) -> list[Any]:
            unflattended_args = tree_unflatten(flat_args[:n_args], args_tree_spec)
            unflattended_kwargs = tree_unflatten(flat_args[n_args:], kwargs_tree_spec)
            return tree_flatten(f(*unflattended_args, **unflattended_kwargs))[0]

        return _wrapper

    return _decorator


def get_argument_flat_device_affinities(
    *args: list[PyTree], **kwargs: dict[str, PyTree]
) -> dict[int, DeviceAffinity]:
    """Return the flat device affinities for unflattened arguments.
    ShardedTensor types have their device affinities assigned.
    All other arguments are left unassigned.

    ```
    get_argument_flat_device_affinities(
        torch.Tensor([1]),
        [ReplicatedTensor(ts=[torch.tensor([2]), torch.tensor([3])])]
    )
    ```
    returns
    ```
    {
        1: DeviceAffinity("0"),
        2: DeviceAffinity("1"),
    }
    ```
    """

    def is_leaf(v: PyTree) -> bool:
        if isinstance(v, ShardedTensor):
            return True
        # TODO: It is sad _is_leaf is private. Find a way not use it.

        return _is_leaf(v)

    # flattened up to a sharded tensor.
    flat_args_up_to_sharded_tensor = tree_flatten((args, kwargs), is_leaf=is_leaf)[0]
    nested_device_affinities: list[list[DeviceAffinity | None]] = [
        [DeviceAffinity(f"{shard_idx}") for shard_idx in range(len(arg.shards))]
        if isinstance(arg, ShardedTensor)
        else [None]
        for arg in flat_args_up_to_sharded_tensor
    ]
    flat_device_affinities: list[DeviceAffinity | None] = [
        affinity
        for affinity_list in nested_device_affinities
        for affinity in affinity_list
    ]
    return {
        arg_idx: affinity
        for arg_idx, affinity in enumerate(flat_device_affinities)
        if affinity is not None
    }


def export(
    f: Callable | None = None,
    fx_builder: FxProgramsBuilder | None = None,
    args: tuple[PyTree] | None = None,
    kwargs: dict[PyTree] | None = None,
    arg_device: dict[int, DeviceAffinity] | None = None,
    *transitive_args,
    **transitive_kwargs,
) -> torch.export.ExportedProgram:
    """Wrapper around FxProgramsBuilder.export_program that handles
    the amdsharktank custom tensor types.

    If `arg_device` is not specified it will extract the affinities
    from the passed `args`.
    `arg_device` must pass the affinities for the flattened arguments.
    These are those that correspond to torch.Tensor.
    For example a sharded tensor with 2 shards would result in 2 arguments in the MLIR
    signature."""

    if f is None:
        return functools.partial(
            export,
            fx_builder=fx_builder,
            args=args,
            kwargs=kwargs,
            arg_device=arg_device,
            *transitive_args,
            **transitive_kwargs,
        )

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if arg_device is None:
        arg_device = get_argument_flat_device_affinities(*args, **kwargs)
    flat_args = tree_flatten((args, kwargs))[0]
    if fx_builder is not None:
        # Flatten the signature of the function.
        # Technically this is done during export, but we want the signature to match
        # the flat device affinities.
        def module_fn_with_flat_signature(module, *flat_args):
            @flatten_signature(*args, **kwargs)
            def flat_fn(*args, **kwargs):
                return f(module, *args, **kwargs)

            return flat_fn(*flat_args)

        amended_kwargs = dict(**transitive_kwargs)
        if "name" not in amended_kwargs or amended_kwargs["name"] is None:
            amended_kwargs["name"] = f.__name__
        return fx_builder.export_program(
            module_fn_with_flat_signature,
            *transitive_args,
            args=tuple(flat_args),
            arg_device=arg_device,
            **amended_kwargs,
        )

    assert False, "TODO: implement the case when not using an FxProgramsBuilder"


def export_model_mlir(
    model: BaseLayer,
    output_path: PathLike,
    *,
    function_batch_sizes_map: Optional[dict[Optional[str], list[int]]] = None,
    batch_sizes: Optional[list[int]] = None,
):
    """Export a model with no dynamic dimensions.

    For the set of provided function name batch sizes pair, the resulting MLIR will
    have function names with the below format.
    ```
    <function_name>_bs<batch_size>
    ```

    If `batch_sizes` is given then it defaults to a single function with named
    "forward".

    The model is required to implement method `sample_inputs`.
    """

    assert not (function_batch_sizes_map is not None and batch_sizes is not None)

    if isinstance(model, ThetaLayer):
        mark_export_external_theta(model.theta)

    if batch_sizes is not None:
        function_batch_sizes_map = {None: batch_sizes}

    if function_batch_sizes_map is None and batch_sizes is None:
        function_batch_sizes_map = {None: batch_sizes}

    fxb = FxProgramsBuilder(model)

    for function, batch_sizes in function_batch_sizes_map.items():
        for batch_size in batch_sizes:
            args, kwargs = model.sample_inputs(batch_size, function)
            dynamic_shapes = model.dynamic_shapes_for_export(
                batch_size=batch_size, function=function
            )

            @fxb.export_program(
                name=f"{function or 'forward'}_bs{batch_size}",
                args=args,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            def _(model, **kwargs):
                return model(**kwargs)

    output = aot.export(fxb)
    output.save_mlir(output_path)
