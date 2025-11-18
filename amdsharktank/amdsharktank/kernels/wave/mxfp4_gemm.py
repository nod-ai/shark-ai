# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.kernels.base import *
from amdsharktank.kernels.mlir_kernel import *
from amdsharktank.kernels.wave.utils import get_wave_module_body_asm, mangle
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    torch_dtype_to_wave,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.compiler.ir import (
    Module,
    Context,
)
import torch


__all__ = [
    "wave_mxfp4_bmm",
]


def wave_mxfp4_batched_gemm(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
    result_torch_dtype: torch.float16,
):
    result_wave_dtype = torch_dtype_to_wave(result_torch_dtype)
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[B, M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, result_wave_dtype],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        casted = tkw.cast(repeat, result_wave_dtype)
        tkw.write(casted, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 128,
        BLOCK_K: 256,
        N: shape[2],
        K: shape[3],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [B, M]
    return batched_gemm, hyperparams, dynamic_symbols


def get_wave_mxfp4_bmm_asm(
    target_function_name: str,
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
    result_torch_dtype: torch.float16,
):
    batched_gemm_func, hyperparams, dynamic_symbols = wave_mxfp4_batched_gemm(
        shape, mfma_variant, enable_scheduling, result_torch_dtype
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        use_buffer_ops=True,
        waves_per_eu=2,
        dynamic_symbols=dynamic_symbols,
        func_name=target_function_name,
        compile_to_mlir=True,
        iree_launch_async=False,
    )
    options = set_default_run_config(options)

    with Context() as ctx:
        batch_size, m, n, k = shape
        batch_size = batch_size if batch_size >= 0 else "B_dyn"
        m = m if m >= 0 else "M_dyn"
        half_k = k // 2
        k_over_thirtytwo = k // 32
        i_type_str = "i8"
        # TODO: don't hardcode the output type, should be dynamic based on the kv-cache-dtype
        o_type_str = "f16"
        kernel_params = {
            B.name: batch_size,
            M.name: m,
            HALF_K.name: half_k,
            K_OVER_THIRTYTWO.name: k_over_thirtytwo,
            N.name: n,
            "input_dtype": i_type_str,
            "output_dtype": o_type_str,
        }
        name = mangle("batched_gemm", **kernel_params)
        batched_gemm_func._name = name
        batched_gemm = wave_compile(options, batched_gemm_func)

    asm = batched_gemm.asm
    return asm


B = DynDim.B
M = DynDim.M
N = StaticDim.N
HALF_K = StaticDim.HALF_K
K_OVER_THIRTYTWO = StaticDim.K_OVER_THIRTYTWO

U8 = Dtype.U8(torch.uint8)
F16 = Dtype.F16(torch.float16)


@mlir_kernel(
    inputs=(
        MLIRTensor[B, M, HALF_K, U8],
        MLIRTensor[B, M, K_OVER_THIRTYTWO, U8],
        MLIRTensor[N, HALF_K, U8],
        MLIRTensor[N, K_OVER_THIRTYTWO, U8],
        MLIRTensor[B, M, N, F16],
    ),
    results=(MLIRTensor[B, M, N, F16],),
)
def wave_mxfp4_bmm(x, x_scales, w_t, w_scales, out, result=None):
    batch_size, m, half_k = x.type.shape
    batch_size, m, k_over_thirtytwo = x_scales.type.shape
    n, half_k = w_t.type.shape
    k = half_k * 2
    shape = (
        batch_size,
        m,
        n,
        k,
    )
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    i_type_str = "i8"
    # TODO: don't hardcode the output type, should be dynamic based on the kv-cache-dtype
    o_type_str = "f16"
    batch_size = batch_size if batch_size >= 0 else "B_dyn"
    m = m if m >= 0 else "M_dyn"
    kernel_params = {
        B.name: batch_size,
        M.name: m,
        HALF_K.name: half_k,
        K_OVER_THIRTYTWO.name: k_over_thirtytwo,
        N.name: n,
        "input_dtype": i_type_str,
        "output_dtype": o_type_str,
    }
    name = mangle("wave_mxfp4_bmm", **kernel_params)
    wave_kernel_fn_name = name

    wave_asm = get_wave_mxfp4_bmm_asm(
        wave_kernel_fn_name, shape, mfma_variant, SchedulingType.PREFETCH, torch.float16
    )

    wave_asm_module = Module.parse(wave_asm)
    wave_asm_body = get_wave_module_body_asm(wave_asm_module)

    mlir_wave_kernel = (
        "\n{% raw %}\n"
        + wave_asm_body
        + "\n{% endraw %}\n"
        + f"""
    util.func private @{{{{kernel_name}}}}(%x : !x, %x_scales : !x_scales, %w_t : !w_t, %w_scales : !w_scales, %out : !out) -> !result {{
        %result = func.call @{wave_kernel_fn_name}(%x, %x_scales, %w_t, %w_scales, %out) : (!x, !x_scales, !w_t, !w_scales, !out) -> !result
        util.return %result : !result
    }}
    """
    )

    mlir = "module {" + mlir_wave_kernel + "}"

    return MLIRSpec(mlir)
