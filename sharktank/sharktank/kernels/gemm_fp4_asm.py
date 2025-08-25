# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *
import torch


M = DynDim.M
N = DynDim.N
K = DynDim.K
HALF_K = DynDim.HALF_K
K_OVER_THIRTYTWO = DynDim.K_OVER_THIRTYTWO

U8 = Dtype.U8(torch.uint8)
F16 = Dtype.F16(torch.float16)
F32 = Dtype.F32(torch.float32)


"""
A4W4 asm gemm kernel
D = A*B*alpha + beta*C

A: [M, K/2] f4x2
B: [N, K/2] f4x2
A_scale: [M, K/32] e8m0 padded
B_scale: [N, K/32] e8m0 padded
bias: [M, N] f32
Out: [M, N] bf16
alpha = 1.0, beta = 0.0 by default
"""


# TODO: Embedding kernels as hex
@mlir_kernel(
    inputs=(
        MLIRTensor[M, HALF_K, U8],
        MLIRTensor[N, HALF_K, U8],
        MLIRTensor[M, K_OVER_THIRTYTWO, U8],
        MLIRTensor[N, K_OVER_THIRTYTWO, U8],
        MLIRTensor[M, N, F32],
    ),
    results=(MLIRTensor[M, N, F16],),
)
def asm_fp4_gemm(x, w, x_scale, w_scale, bias, result=None):
    mlir = f"""
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {{target_arch = "gfx950", ukernels = "none"}}>
module {{
{{% raw %}}
    util.func private @asm_mxfp4_gemm(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?x?xi8>, %arg3: tensor<?x?xi8>, %arg4: tensor<?x?xf32>) -> (tensor<?x?xf16>) {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c255 = arith.constant 255 : index
        %c256 = arith.constant 256 : index
        %m = tensor.dim %arg0, %c0 : tensor<?x?xi8>
        %n = tensor.dim %arg1, %c0 : tensor<?x?xi8>
        %k_f4x2 = tensor.dim %arg0, %c1 : tensor<?x?xi8>
        %k_e8m0 = tensor.dim %arg2, %c1 : tensor<?x?xi8>
        %k = arith.muli %k_f4x2, %c2 : index
        // m_256 = (m + 255) // 256 * 256
        %m_256 = affine.apply affine_map<()[s0] -> (s0 ceildiv 256 * 256)>()[%m]
        %hi_pad = arith.subi %m_256, %m : index
        %c0_i8 = arith.constant 1 : i8
        %x_scales_padded = tensor.pad %arg2 low[%c0, %c0] high[%hi_pad, %c0] {{
        ^bb0(%i0: index, %i1: index):
            tensor.yield %c0_i8 : i8
        }} : tensor<?x?xi8> to tensor<?x?xi8>
        %c0_f32 = arith.constant 1.0 : f32
        %bias_padded = tensor.pad %arg4 low[%c0, %c0] high[%hi_pad, %c0] {{
        ^bb0(%i0: index, %i1: index):
            tensor.yield %c0_f32 : f32
        }} : tensor<?x?xf32> to tensor<?x?xf32>
        %alpha = arith.constant 1.0 : f32
        %beta = arith.constant 0.0 : f32
        %alpha_i32 = arith.bitcast %alpha : f32 to i32
        %beta_i32  = arith.bitcast %beta  : f32 to i32
        %m_i32 = arith.index_cast %m : index to i32
        %n_i32 = arith.index_cast %n : index to i32
        %k_i32 = arith.index_cast %k : index to i32
        %k_e8m0_i32 = arith.index_cast %k_e8m0 : index to i32
        %gemm = hal.dispatch.extern "f4gemm_kernel_func"[%m, %n](%alpha_i32, %beta_i32, %k_i32, %k_i32, %n_i32, %m_i32, %n_i32, %k_i32, %k_e8m0_i32, %k_e8m0_i32, %arg0, %arg1, %x_scales_padded, %arg3, %bias_padded) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, tensor<?x?xi8>{{%m, %k_f4x2}}, tensor<?x?xi8>{{%n, %k_f4x2}}, tensor<?x?xi8>{{%m_256, %k_e8m0}}, tensor<?x?xi8>{{%n, %k_e8m0}}, tensor<?x?xf32>{{%m_256, %n}}) -> tensor<?x?xbf16>{{%m_256, %n}}
            count(%device: !hal.device, %m_workload: index, %n_workload: index) -> (index, index, index) {{
                %c1_0 = arith.constant 1 : index
                %subm = arith.constant 256 : index
                %subn = arith.constant 256 : index
                // gdx = (Ndim + SUBN - 1) / SUBN
                // gdy = (Mdim + SUBM - 1) / SUBM
                %subn_sub1 = arith.subi %subn, %c1_0 : index
                %n_add = arith.addi %n_workload, %subn_sub1 : index
                %gdx = arith.divui %n_add, %subn : index
                %subm_sub1 = arith.subi %subm, %c1_0 : index
                %m_add = arith.addi %m_workload, %subm_sub1 : index
                %gdy = arith.divui %m_add, %subm : index
                %gdz = arith.constant 1 : index
                hal.return %gdx, %gdy, %gdz : index, index, index
            }}
            layout(#hal.pipeline.layout<constants = 10, bindings = [
                #hal.pipeline.binding<storage_buffer, ReadOnly>,
                #hal.pipeline.binding<storage_buffer, ReadOnly>,
                #hal.pipeline.binding<storage_buffer, ReadOnly>,
                #hal.pipeline.binding<storage_buffer, ReadOnly>,
                #hal.pipeline.binding<storage_buffer, ReadOnly>,
                #hal.pipeline.binding<storage_buffer>
            ]>)
            objects({{
                #rocm_target ordinal(0) = [
                    #hal.executable.object<{{
                        path = "sharktank/sharktank/kernels/compiled_kernels/f4gemm_outBF16_tn_256x256_scale.co"
                    }}>
                ]
            }})
            attributes {{subgroupSize = 64 : i64, workgroup_size = [256 : index, 1 : index, 1 : index]}}
        %gemm_slice = tensor.extract_slice %gemm[0, 0] [%m, %n] [1, 1] : tensor<?x?xbf16> to tensor<?x?xbf16>
        %out_init = tensor.empty(%m, %n) : tensor<?x?xf16>
        %gemm_f16 = linalg.generic {{indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"]}} ins(%gemm_slice : tensor<?x?xbf16>) outs(%out_init : tensor<?x?xf16>) {{
        ^bb0(%in: bf16, %out: f16):
            %in_f32 = arith.extf %in : bf16 to f32
            %in_f16 = arith.truncf %in_f32 : f32 to f16
            linalg.yield %in_f16 : f16
        }} -> tensor<?x?xf16>
        util.return %gemm_f16 : tensor<?x?xf16>
    }}
    util.func private @shuffle_scales(%arg0: tensor<?x?xi8>) -> tensor<?x?xi8> {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
        %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xi8>
        %dim0_i32 = arith.index_cast %dim0 : index to i32
        %dim1_i32 = arith.index_cast %dim1 : index to i32
        %MXFP4_QUANT_BLOCK_SIZE = arith.constant 32 : i32
        %N = arith.muli %dim1_i32, %MXFP4_QUANT_BLOCK_SIZE : i32
        %scaleM_index = affine.apply affine_map<()[s0] -> (s0 ceildiv 32 * 32)>()[%dim0]
        %scaleM = arith.index_cast %scaleM_index : index to i32
        // Note: This is not safe if the dim size exceeds INT32_MAX. To pass a 64
        // bit value it must be broken down into two 32-bit values for the high and
        // low bits.
        // %dim_i32 = arith.index_cast %dim : index to i32
        // Inline external dispatch that conforms to the ABI that the kernel
        // requires. This is the primary reason for the surrounding function as
        // details like tensor shape and push constants need to line up after
        // splicing in the custom dispatch. This allows the kernel author to manage
        // such details by hand without needing the rewrite patterns to worry about
        // things like order of push constants.
        // arg6 = scaleN_pad
        // arg5 = scaleM_pad
        // arg4 = N
        // arg3 = M
        // arg2 = stride_M
        // arg1 = output
        // arg0 = input
        %4 = hal.dispatch.extern "_mxfp4_quant_shuffle"[%dim0, %dim1](%dim1_i32, %dim0_i32, %N, %scaleM, %dim1_i32, %arg0) : (i32, i32, i32, i32, i32, tensor<?x?xi8>{{%dim0, %dim1}}) -> tensor<?x?xi8>{{%dim0, %dim1}}
            count(%device: !hal.device, %dim_m: index,  %dim_n: index) -> (index, index, index) {{
                %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%dim_m]
                %c1_0 = arith.constant 1 : index
                hal.return %x, %dim_n, %c1_0 : index, index, index
            }}
            layout(#hal.pipeline.layout<constants = 5, bindings = [
                #hal.pipeline.binding<storage_buffer, ReadOnly>,
                #hal.pipeline.binding<storage_buffer>
            ]>)
            objects({{
                #rocm_target ordinal(0) = [
                    #hal.executable.object<{{
                        path = "sharktank/sharktank/kernels/compiled_kernels/mxfp4_quant_shuffle.co"
                    }}>
                ]
            }})
            attributes {{subgroupSize = 64, workgroup_size = [128 : index, 1 : index, 1 : index]}}
        util.return %4 : tensor<?x?xi8>
    }}
{{% endraw %}}
    util.func private @{{{{kernel_name}}}}(%x: !x, %w: !w, %x_scale: !x_scale, %w_scale: !w_scale, %bias: !bias) -> !result {{
        %x_scale_shuffle = util.call @shuffle_scales(%x_scale) : (!x_scale) -> !x_scale
        %w_scale_shuffle = util.call @shuffle_scales(%w_scale) : (!w_scale) -> !w_scale
        %result = util.call @asm_mxfp4_gemm(%x, %w, %x_scale_shuffle, %w_scale_shuffle, %bias) : (!x, !w, !x_scale, !x_scale, !bias) -> !result
        util.return %result : !result
    }}
}}
        """

    return MLIRSpec(mlir)
