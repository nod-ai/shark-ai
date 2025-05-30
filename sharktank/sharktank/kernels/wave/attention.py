# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_bhsd_attention_kernel,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.compiler.ir import (
    ArrayAttr,
    Attribute as Attribute,
    Block,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    DictAttr,
    FloatAttr,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    Float8E4M3FNType,
    Float8E5M2FNUZType,
    Float8E5M2Type,
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    TypeAttr,
    IntegerAttr,
    IntegerType,
    MLIRError,
    RankedTensorType,
    Location,
    Module,
    Operation,
    StringAttr,
    SymbolTable,
    Type as IrType,
    Value,
)
from iree.compiler.dialects import builtin, func, util
from iree.turbine.transforms.merger import Merger

__all__ = [
    "build_util_func",
    "get_wave_flash_attention_asm"
    "wave_bhsd_flash_attention",
]


def build_util_func(kernel_name: str, wave_kernel_name: str, *operands: Value):
    with Context() as ctx, Location.unknown():
        module = Module.create()

        operand_types = [op.type for op in operands if op is not None]
        result_type = operand_types[-1]
        func_type = FunctionType.get(
            inputs=operand_types,
            results=[result_type]
        )

        with InsertionPoint(module.body):
            # Manually build util.func using Operation.create
            sym_name = StringAttr.get(kernel_name)
            type_attr = TypeAttr.get(func_type)
            visibility_attr = StringAttr.get("private")

            # Create the util.func operation
            util_func_op = Operation.create(
                "func.func",
                results=[],
                operands=[],
                attributes={
                    "sym_name": sym_name,
                    "function_type": type_attr,
                    "sym_visibility": visibility_attr
                },
                regions=1
            )

            # Insert and populate the body block
            module.body.append(util_func_op)
            block = util_func_op.regions[0].blocks.append(*func_type.inputs)

            with InsertionPoint(block):
                call_op = func.CallOp(
                    func_type.results,
                    FlatSymbolRefAttr.get(wave_kernel_name),
                    block.arguments
                )
                func.ReturnOp([call_op.result])
        return module


def get_wave_flash_attention_asm(target_function_name: str, shape: AttentionShape, mfma_variant: list[MMAType], dynamic_dims: bool, is_causal: bool = False, is_custom_mask: bool = False,) -> str:
    (
        base_attention_func,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_bhsd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=is_causal,
        is_custom_mask=is_custom_mask,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        func_name=target_function_name,
        compile_to_mlir=True,
    )
    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention_func)

    asm = base_attention.asm
    return asm

# Wave Attention Kernels
# Each kernel is put into its own class to create a namespace for it
B = StaticDim.B # batch_size
H = StaticDim.H # num_query_heads
M = StaticDim.M # query_seq_len
N = StaticDim.N # head_size_kv
K1 = StaticDim.K1 # head_size
K2 = StaticDim.K2 # kv_seq_len

F16 = Dtype.F16
F32 = Dtype.F32

@mlir_kernel(
    inputs=(
        MLIRTensor[
            B,
            H,
            M,
            K1,
            F16
        ],
        MLIRTensor[
            B,
            H,
            K2,
            K1,
            F16
        ],
        MLIRTensor[
            B,
            H,
            K2,
            N,
            F16
        ],
        MLIRTensor[
            B,
            H,
            M,
            N,
            F32
        ],
    ),
    results=(
        MLIRTensor[B, H, M, N, F32],
    ),
)
def wave_bhsd_flash_attention(
    q, k, v, c, result=None
):
    batch_size, num_heads, q_s, q_d = q.type.shape
    v_batch_size, num_heads_kv, v_s, v_d = v.type.shape
    shape = AttentionShape(
        batch_size=batch_size,
        num_query_heads=num_heads,
        num_kv_heads=num_heads_kv,
        query_seq_len=q_s,
        head_size_kv=v_d,
        head_size=q_d,
        kv_seq_len=v_s,
    )
    mfma_variant = (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16)
    dynamic_dims = False
    is_causal = True
    is_custom_mask = False
    i_type_str = "f16"
    o_type_str = "f32"

    wave_kernel_name = f"wave_flash_attention_{batch_size}_{num_heads}_{q_s}_{v_d}_{i_type_str}_{o_type_str}"

    asm = get_wave_flash_attention_asm(
        wave_kernel_name,
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=is_causal,
        is_custom_mask=is_custom_mask,
    )

    asm_op = Operation.parse(asm)
    op = build_util_func("{{kernel_name}}", wave_kernel_name, q, k, v, c)
    merger = Merger(
        op.operation, asm_op.operation
    )
    merger.merge()
    mlir = asm_op.operation.get_asm()
    with open('test.mlir', 'w') as f:
        f.write(mlir)

    return MLIRSpec(mlir)
