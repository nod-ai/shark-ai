# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *

import torch

from iree.turbine.support.ir_imports import (
    ComplexType,
    F16Type,
    F32Type,
    RankedTensorType,
    ShapedType,
    Value,
    flow_d,
    tensor_d,
)

from iree.turbine.runtime.op_reg import (
    CustomOp,
    KernelBuilder,
    KernelSelection,
)

__all__ = [
    "bitcast_to_complex",
    "bitcast_to_real",
]

_ftype_to_ctype_table = {
    torch.bfloat16: torch.complex32,
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
}

_ctype_to_ftype_table = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
}

_type_to_irtype_table = {
    torch.float16: lambda: F16Type.get(),
    torch.float32: lambda: F32Type.get(),
    torch.complex32: lambda: ComplexType.get(F16Type.get()),
    torch.complex64: lambda: ComplexType.get(F32Type.get()),
}


@CustomOp.register(library=LIBRARY)
class bitcast_to_complex(CustomOp):

    signature = "bitcast_to_complex(Tensor q) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        ta = ksel.arg_tensor(0)

        torch._check(ta.t.dtype in _ftype_to_ctype_table)
        torch._check(isinstance(ta.t.shape[-1], int))

        new_shape = [i for i in ta.t.shape]
        new_shape[-1] = new_shape[-1] // 2

        ctype = _ftype_to_ctype_table[ta.t.dtype]
        ret = ksel.return_new_tensor(new_shape, dtype=ctype)
        specialize_all_known_dims(ta)
        specialize_all_known_dims(ret)

    def eager_execute(self, tensor):
        return torch.view_as_complex(tensor.unflatten(-1, (-1, 2)))

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        t = kb.arg_bindings[0]
        result_desc = ksel.result_descs[0]
        result_shape = [
            d if isinstance(d, int) else RankedTensorType.get_dynamic_size()
            for d in result_desc.t.shape
        ]

        dynamic_dims: list[Value] = []
        _append_dynamic_dims(kb, dynamic_dims, t)

        c64 = _type_to_irtype_table[result_desc.t.dtype]()
        rtt = RankedTensorType.get(result_shape, c64)
        result = flow_d.TensorBitCastOp(rtt, t, dynamic_dims, dynamic_dims).result
        kb.yield_results(result)


@CustomOp.register(library=LIBRARY)
class bitcast_to_real(CustomOp):

    signature = "bitcast_to_real(Tensor q) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        ta = ksel.arg_tensor(0)

        torch._check(ta.t.dtype in _ctype_to_ftype_table)
        torch._check(isinstance(ta.t.shape[-1], int))

        new_shape = [i for i in ta.t.shape]
        new_shape[-1] = new_shape[-1] * 2

        ftype = _ctype_to_ftype_table[ta.t.dtype]
        ret = ksel.return_new_tensor(new_shape, dtype=ftype)
        specialize_all_known_dims(ta)
        specialize_all_known_dims(ret)

    def eager_execute(self, tensor):
        return torch.view_as_real(tensor).flatten(-2, -1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        t = kb.arg_bindings[0]
        result_desc = ksel.result_descs[0]
        result_shape = [
            d if isinstance(d, int) else RankedTensorType.get_dynamic_size()
            for d in result_desc.t.shape
        ]

        dynamic_dims: list[Value] = []
        _append_dynamic_dims(kb, dynamic_dims, t)

        ftype = _type_to_irtype_table[result_desc.t.dtype]()
        rtt = RankedTensorType.get(result_shape, ftype)
        result = flow_d.TensorBitCastOp(rtt, t, dynamic_dims, dynamic_dims).result
        kb.yield_results(result)


################################################################################
# Emission utilities
################################################################################


def _append_dynamic_dims(kb: KernelBuilder, dynamic_dims: list[Value], tensor: Value):
    rtt = RankedTensorType(tensor.type)
    for i in range(rtt.rank):
        if rtt.is_dynamic_dim(i):
            dynamic_dims.append(tensor_d.dim(tensor, kb.constant_index(i)))
