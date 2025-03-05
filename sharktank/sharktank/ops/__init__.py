# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This package contains custom operation-like functions which operate on a mix
of `torch.Tensor` and `InferenceTensor` type hierarchies. Available ops
are defined in `signatures`. Specific implementations are in `_impl` modules.

There is a simple `_registry` which allows multiple implementations to be
registered against a signature. Registration is done by type signature. Any
matching implementations are processed in reverse (salience, def order) order.
The first one that does not return NotImplemented is used.

In this way, core operations can be defined over a mix of tensor-like types
and layouts.
"""

from . import _registry
from ..types.tensors import unbox_tensor

def import_and_wrap_signatures():
    """
    Import the signatures from .signatures and wrap then so the shards of their inputs tensors are on the same devices.
    For unary ops, also pins the result if the input is pinned (e.g. for transpose).
    """
    def transfer_n_pin(f):
        """
        Create a wrapper for each operation.
        """
        from ..types import ShardedTensor
        from typing import List, Tuple, Dict, Any
        def unwrap_args(items: Tuple | Dict[str, Any]) -> Tuple[List[int | List[int]], List[ShardedTensor]]:
            t_i, t_vals = [], []
            for i, arg in enumerate(items):
                if isinstance(arg, ShardedTensor):
                    t_i.append(i)
                    t_vals.append(arg)
                elif isinstance(arg, list) and all(isinstance(val, ShardedTensor) for val in arg):
                    t_i.append([i] * len(arg))
                    t_vals.extend(arg)
            return t_i, t_vals

        def rewrap_args(items: Tuple | Dict, t_i: List[int | List[int]], t_vals: List[ShardedTensor]) -> Tuple[Tuple, Dict[str, Any]]:
            i_lookup = list(range(len(items))) if isinstance(items, tuple) else list(items.keys())
            new_items = list(items) if isinstance(items, tuple) else dict(items)

            for i in t_i:
                if isinstance(i, int):
                    new_items[i_lookup[i]] = t_vals.pop(0)
                else:  # List[int]
                    _popped_vals = [t_vals.pop(0) for _ in range(len(i))]
                    new_items[i_lookup[i[0]]] = items[i_lookup[i[0]]].__class__(_popped_vals)

            if isinstance(new_items, list):
                new_items = tuple(new_items)
            return new_items

        def func_wrapper(*args: Tuple, **kwargs: Dict[str, Any]):
            t_i_args, t_vals_args = unwrap_args(args)
            t_i_kwargs, t_vals_kwargs = unwrap_args(list(kwargs.values()))
            t_vals = t_vals_args + t_vals_kwargs

            t_vals = transfer_if_needed(*t_vals)

            args = rewrap_args(args, t_i_args, t_vals[:len(t_vals_args)])
            kwargs = rewrap_args(kwargs, t_i_kwargs, t_vals[len(t_vals_args):])
            res = f(*args, **kwargs)
            if isinstance(res, ShardedTensor) and len(t_vals) > 0:
                pinned = res.pinned or (len(t_vals) == 1 and t_vals[0].pinned)
                res = res.clone(devices=t_vals[0].devices, pinned=pinned)
            return res
        
        if hasattr(f, "override"):  # Needed for ops like gelu_tanh_approximation
            func_wrapper.override = f.override
        return func_wrapper

    do_not_wrap = {'all_gather', 'all_reduce', 'replicate', 'index_copy_', 'index_put_'}

    from . import signatures
    for func_name in signatures.__all__:
        func = getattr(signatures, func_name)
        if func_name not in do_not_wrap:
            func = transfer_n_pin(func)
        globals()[func_name] = func
import_and_wrap_signatures()

from .shape import *

# Ensure that implementations are registered.
# Note that delegation prefers matching ops defined later, so order here
# can be important.
from . import default_impls
from . import custom_impls
from . import sharded_impls

from . import attention_impls

# Comment this out to completely disable optimized quantized implementations.
from . import qconv_impls
from . import qlinear_impls

from .sharded_impls import transfer_if_needed  # TODO: Hack just to get tests running, figure out properly later