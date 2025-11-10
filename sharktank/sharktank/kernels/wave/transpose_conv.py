# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel._support.dtype import DataType
from typing import Any, Optional
from iree.turbine.kernel.lang.global_symbols import (
    SHARED_ADDRESS_SPACE,
    GLOBAL_ADDRESS_SPACE,
)
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.compiler.ir import (
    Module,
    Context,
)
from sharktank.kernels.mlir_kernel import *
import torch
from sharktank.kernels.wave.utils import get_wave_module_body_asm


def transpose_conv2d(
    layout: str,
    n: int,
    h: int,
    w: int,
    c: int,
    hf: int,
    wf: int,
    nf: int,
    upsamp_stride: int,
    conv_stride: int,
    input_dtype: DataType,
    output_dtype: DataType,
    mem_space: tkl.IndexSymbol = SHARED_ADDRESS_SPACE,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    ratio_m: Optional[int] = None,
    ratio_n: Optional[int] = None,
) -> tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]:
    """This Kernel computes the transpose convolution with a set upsampling stride. An upsample stride of 1 leaves the input matrix in its orginal form. The weight matrix is flipped in both spatial dims and the input matrix is upsampled before doing the convolution process.

    Parameters:
        layout (str): Either "nchw_fchw" or "nhwc_hwcf" based on the ordering of the dims of the input tensors.
        n (int): Number of input  (Batch size).
        h (int): Height of input matrix.
        w (int): Width of input matrix.
        c (int): Number of channels in input matrix.
        hf (int): Height of weight (filter) matrix.
        wf (int): Width of weight (filter) matrix.
        nf (int): Number of filters.
        upsamp_stride (int): Stride to determine placement of 0 row/col inserts, upsamp_stride = 1 is no upsampling.
        conv_stride (int): Convolution stride distance.
        input_dtype (DataType): Input and filter datatype, currently only supports tkl.f16.
        output_dtype (DataType): Output matrix datatype, currently only supports tkl.f32.
        mem_space: tkl.IndexSymbol = SHARED_ADDRESS_SPACE,
        block_m (int | None): M dim tile size.
        block_n (int | None): N dim tile size.
        block_k (int | None): K dim tile size.
        ratio_m (int | None): Divider for M dim tile size and number of waves.
        ratio_n (int | None): Divider for N dim tile size and number of waves.

    Returns:
        output (tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]): Wave kernel to be compiled and hyperparameters.
    """

    # Input Checks
    assert input_dtype == tkl.f16, f"Unsupported input dtype: {input_dtype}"
    assert output_dtype == tkl.f32, f"Unsupported input dtype: {output_dtype}"
    if h != w:
        raise ValueError(f"Input matrix must have square spatial dims, {h=} and {w=}")
    if hf != wf:
        raise ValueError(
            f"Weight Matrix must have square spatial dims, {hf=} and {wf=}"
        )
    if upsamp_stride < 1:
        raise ValueError(
            f"upsamp_stride must be >= 1 not {upsamp_stride}, for no stride use upsamp_stride = 1"
        )

    padding = 0  # only pad=0 is supported for now

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF
    UPSAMP_STRIDE = sym.UPSAMP_STRIDE

    H_OUT = (H * upsamp_stride + 2 * padding - HF) // conv_stride + 1
    W_OUT = (W * upsamp_stride + 2 * padding - WF) // conv_stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    # Uses iGEMM method for flattening 4D to 2D matrix for efficent MM and implicitly upsampling during iGEMM
    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j % C,
            H: ((i % SZ_OUT) % W_OUT * conv_stride + (j // C) % WF) // UPSAMP_STRIDE,
            W: ((i % SZ_OUT) // W_OUT * conv_stride + (j // C) // WF) // UPSAMP_STRIDE,
        },
        outputs={M: i, K: j},
    )

    # Flips weight matrix across spatial dims in index mapping
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            NF: i % NF,
            C: j % C,
            HF: HF - 1 - (j // C) % WF,
            WF: WF - 1 - (j // C) // WF,
        },
        outputs={NF: i, K: j},
    )

    # Map flattened conv result back into N, NF, H, W
    out_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, NF: j},
        outputs={
            N: i // SZ_OUT,
            NF: j,
            H_OUT: (i % SZ_OUT) % W_OUT,
            W_OUT: (i % SZ_OUT) // W_OUT,
        },
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    if layout == "nchw_fchw":
        x_type = tkl.Memory[N, C, H, W, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, NF, H_OUT, W_OUT, GLOBAL_ADDRESS_SPACE, output_dtype]
    elif layout == "nhwc_hwcf":
        x_type = tkl.Memory[N, H, W, C, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[HF, WF, C, NF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, H_OUT, W_OUT, NF, GLOBAL_ADDRESS_SPACE, output_dtype]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if block_m is None:
        block_m = 64

    if block_n is None:
        block_n = 128

    if block_k is None:
        block_k = 32

    if ratio_m is None:
        ratio_m = 2

    if ratio_n is None:
        ratio_n = 2

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []

    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N / ratio_n)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(ratio_m, ratio_n, 1),
        )
    ]

    @tkw.wave(constraints)
    def trans_conv(
        x: x_type,
        we: we_type,
        upsamp_stride: tkl.i32,
        out: out_type,
    ):
        # Set upsamp symbol with upsamp_stride value
        tkw.set_symbol(UPSAMP_STRIDE, upsamp_stride)

        # Create reduction loop Register
        c_reg = tkl.Register[M, NF, output_dtype](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, NF, output_dtype],
        ) -> tkl.Register[M, NF, output_dtype]:
            # Read input matrix
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )

            # Get i and j indx
            i_idx = tkw.self_index(M, tkl.i32)
            j_idx = tkw.self_index(K, tkl.i32)

            # Broadcast indices to M, K shape
            i_idx = tkw.broadcast(i_idx, target_shape=[M, K])
            j_idx = tkw.broadcast(j_idx, target_shape=[M, K])

            # Mask function for computing valid indices of MxK matrix
            mask_func = lambda i, j: (
                ((((i % SZ_OUT) % W_OUT) + (j // C) % wf) % UPSAMP_STRIDE)
                + (((((i % SZ_OUT) // W_OUT) + (j // C) // wf) % UPSAMP_STRIDE))
                < 1
            )

            # Create mask and cast to f16
            mask = tkw.apply_expr((i_idx, j_idx), mask_func)
            mask = tkw.broadcast(mask, target_shape=[M, K])
            mask = tkw.cast(mask, input_dtype)

            # Apply mask to input matrix
            a_reg = a_reg * mask

            # Read filter
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )

            # compute mma for MxK x KxNF
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Write result using unflattening output mapping
        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=ELEMS_PER_THREAD
        )

    symbols = {
        N: n,
        C: c,
        W: w,
        H: h,
        NF: nf,
        WF: wf,
        HF: hf,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: mem_space,
    }

    return trans_conv, symbols


def get_transpose_conv_asm(
    target_function_name: str,
    layout: str,
    n: int,
    h: int,
    w: int,
    c: int,
    hf: int,
    wf: int,
    nf: int,
    upsamp_stride: int,
    conv_stride: int,
    input_dtype: DataType,
    output_dtype: DataType,
    mem_space: tkl.IndexSymbol,
):

    transpose_conv_func, hyperparams = transpose_conv2d(
        layout,
        n,
        h,
        w,
        c,
        hf,
        wf,
        nf,
        upsamp_stride,
        conv_stride,
        input_dtype,
        output_dtype,
        mem_space,
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        use_stride_cache_swizzle=True,
        compile_to_mlir=True,
        func_name=target_function_name,
    )

    options = set_default_run_config(options)

    with Context() as ctx:
        transpose_conv_func._name = "trans_conv"
        transpose_conv_func = wave_compile(options, transpose_conv_func)

    asm = transpose_conv_func.asm
    return asm


N = StaticDim.N
C = StaticDim.C
H = StaticDim.H
W = StaticDim.W
NF = StaticDim.NF
WF = StaticDim.WF
HF = StaticDim.HF
H_OUT = StaticDim.H_OUT
W_OUT = StaticDim.W_OUT
UPSAMP_STRIDE = StaticDim.UPSAMP_STRIDE

U32 = Dtype.U32(torch.int32)
F16 = Dtype.F16(torch.float16)
F32 = Dtype.F32(torch.float32)


@mlir_kernel(
    inputs=(
        MLIRTensor[N, C, H, W, F16],
        MLIRTensor[NF, C, HF, WF, F16],
        MLIRTensor[N, NF, H_OUT, W_OUT, F32],
        MLIRTensor[U32],
    ),
    results=(MLIRTensor[N, NF, H_OUT, W_OUT, F32],),
)
def wave_transpose_conv(x, we, out, upsamp_stride, result=None):

    layout = "nchw_fchw"
    upsamp_stride_val = 2  # TODO: Not how to get actual value from input

    n, c, h, w = x.type.shape
    (
        nf,
        _,
        hf,
        wf,
    ) = we.type.shape
    conv_stride = 1
    mem_space = SHARED_ADDRESS_SPACE
    wave_kernel_name = f"wave_trans_conv_n_{n}_c_{c}_h_{h}_w_{w}_nf_{nf}_cf_{c}_hf_{hf}_wf_{wf}_upStride_{upsamp_stride_val}"

    wave_asm = get_transpose_conv_asm(
        target_function_name=wave_kernel_name,
        layout=layout,
        n=n,
        h=h,
        w=w,
        c=c,
        hf=hf,
        wf=wf,
        nf=nf,
        upsamp_stride=upsamp_stride_val,
        conv_stride=conv_stride,
        input_dtype=tkl.f16,
        output_dtype=tkl.f32,
        mem_space=mem_space,
    )

    wave_asm_module = Module.parse(wave_asm)
    wave_asm_body = get_wave_module_body_asm(wave_asm_module)
    mlir_wave_kernel = (
        "\n{% raw %}\n"
        + wave_asm_body
        + "\n{% endraw %}\n"
        + f"""
    util.func private @{{{{kernel_name}}}}(%arg0: !x, %arg1: !we, %arg2: !out, %arg3: !upsamp_stride) -> !result {{
    %upsamp_i32 = tensor.extract %arg3[] : tensor<i32>
    %result = func.call @{wave_kernel_name}(%arg0, %arg1, %arg2, %upsamp_i32)
                : (!x, !we, !out, i32) -> !result
    util.return %result : !result
    }}
    """
    )

    mlir = "module {" + mlir_wave_kernel + "}"
    return MLIRSpec(mlir)
