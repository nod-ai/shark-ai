# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *
from sharktank.utils.logging import get_logger
from sharktank.kernels.assembly_binaries import *
import torch

logger = get_logger(__name__)

M = DynDim.M
M_PADDED = DynDim.M_PADDED
N = DynDim.N
K = DynDim.K
HALF_K = DynDim.HALF_K
K_OVER_THIRTYTWO = DynDim.K_OVER_THIRTYTWO

U8 = Dtype.U8(torch.uint8)
F16 = Dtype.F16(torch.float16)
F32 = Dtype.F32(torch.float32)


def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_.view(x_type)


def shuffle_scale(x: torch.Tensor) -> torch.Tensor:
    """Shuffle scale tensors according to the scales_shuffle_encoding layout.

    This implements the swizzle pattern from the MLIR encoding:
    - innerTileSizes = [32, 8]
    - expandShape = [[CrossIntrinsic: 2, Internal: 16], [CrossIntrinsic: 2, Internal: 4]]
    - permutation = [3, 1, 2, 0]

    Scale shape is typically [N, K/32] where each element is a scale for a 32-element block.
    """
    x_type = x.dtype
    if x_type == torch.uint8:
        # Scales are already uint8, no conversion needed
        pass
    else:
        # Convert to uint8 view if needed
        x = x.view(torch.uint8)

    # Tile sizes for scale shuffling
    TILE_M = 32  # innerTileSizes[0]
    TILE_K = 8  # innerTileSizes[1]

    # expandShape dimensions
    # Dim 0: [CrossIntrinsic: 2, Internal: 16] = 2 * 16 = 32
    # Dim 1: [CrossIntrinsic: 2, Internal: 4] = 2 * 4 = 8
    CROSS_M = 2
    INTERNAL_M = 16
    CROSS_K = 2
    INTERNAL_K = 4

    assert (
        x.shape[-2] % TILE_M == 0
    ), f"Dim -2 ({x.shape[-2]}) must be divisible by {TILE_M}"
    assert (
        x.shape[-1] % TILE_K == 0
    ), f"Dim -1 ({x.shape[-1]}) must be divisible by {TILE_K}"

    # Reshape to expose the tile structure
    # [..., M, K] -> [..., M//32, 2, 16, K//8, 2, 4]
    #                      [   0,  1,  2,   3, 4, 5]
    x_ = x.view(
        -1,
        x.shape[-2] // TILE_M,
        CROSS_M,
        INTERNAL_M,
        x.shape[-1] // TILE_K,
        CROSS_K,
        INTERNAL_K,
    )

    # Apply permutation [3, 1, 2, 0]
    # This maps the last 4 dimensions: [1, 2, 3, 4, 5, 6] indices relative to expanded dims
    # The permutation [3, 1, 2, 0] refers to the expanded dimensions:
    # [CROSS_M(1), INTERNAL_M(2), CROSS_K(4), INTERNAL_K(5)]
    # New order: [INTERNAL_K(5), CROSS_M(1), INTERNAL_M(2), CROSS_K(4)]
    x_ = x_.permute(
        0, 1, 5, 2, 4, 3, 6
    )  # batch, M_tiles, INTERNAL_K, CROSS_M, K_tiles, INTERNAL_M, CROSS_K
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)

    return x_.view(x_type)


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


def _build_mlir_spec(bitstring, shuffle_scales=True):
    """Helper function to build MLIR spec with the given bitstring.

    Args:
        bitstring: The kernel binary data
        shuffle_scales: If True, shuffle scale. Should only be False when scales are preshuffled.
    """

    mlir = f"""
#rocm_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {{target_arch = "gfx950", ukernels = "none"}}>
#scales_shuffle_encoding = #iree_encoding.layout<[#iree_gpu.gpu_encoding_resolver<configuration =
    {{encoding_info = {{
        innerDimsPos = [0, 1],
        innerTileSizes = [32, 8],
        outerDimsPerm = [0, 1],
    swizzle = {{
        expandShape = [
        [["CrossIntrinsic", 2 : i16], ["Internal", 16 : i16]],
        [["CrossIntrinsic", 2 : i16], ["Internal", 4 : i16]] ],
        permutation = [3, 1, 2, 0]
}}}}}}>]>
module {{
{{% raw %}}
    util.func private @asm_mxfp4_gemm(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?x?xi8, #scales_shuffle_encoding>, %arg3: tensor<?x?xi8, #scales_shuffle_encoding>, %arg4: tensor<?x?xf32>) -> (tensor<?x?xf16>) {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c32 = arith.constant 32 : index
        %c255 = arith.constant 255 : index
        %c256 = arith.constant 256 : index
        %m = tensor.dim %arg0, %c0 : tensor<?x?xi8>
        %m_padded = tensor.dim %arg4, %c0 : tensor<?x?xf32>
        %n = tensor.dim %arg1, %c0 : tensor<?x?xi8>
        %k_f4x2 = tensor.dim %arg0, %c1 : tensor<?x?xi8>
        %k = arith.muli %k_f4x2, %c2 : index
        %k_e8m0 = arith.divui %k, %c32 : index
        %alpha = arith.constant 1.0 : f32
        %beta = arith.constant 0.0 : f32
        %alpha_i32 = arith.bitcast %alpha : f32 to i32
        %beta_i32  = arith.bitcast %beta  : f32 to i32
        %m_i32 = arith.index_cast %m : index to i32
        %n_i32 = arith.index_cast %n : index to i32
        %k_i32 = arith.index_cast %k : index to i32
        %k_e8m0_i32 = arith.index_cast %k_e8m0 : index to i32
        %gemm = hal.dispatch.extern "f4gemm_kernel_func"[%m, %n](%alpha_i32, %beta_i32, %k_i32, %k_i32, %n_i32, %m_i32, %n_i32, %k_i32, %k_e8m0_i32, %k_e8m0_i32, %arg0, %arg1, %arg2, %arg3, %arg4) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, tensor<?x?xi8>{{%m, %k_f4x2}}, tensor<?x?xi8>{{%n, %k_f4x2}}, tensor<?x?xi8, #scales_shuffle_encoding>{{%m_padded, %k_e8m0}}, tensor<?x?xi8, #scales_shuffle_encoding>{{%n, %k_e8m0}}, tensor<?x?xf32>{{%m_padded, %n}}) -> tensor<?x?xbf16>{{%m_padded, %n}}
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
                        path = "",
                        data = {bitstring}
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
    util.func private @shuffle_scales(%arg0: tensor<?x?xi8>) -> tensor<?x?xi8, #scales_shuffle_encoding> {{
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
        %4 = iree_encoding.set_encoding %arg0 : tensor<?x?xi8>
            -> tensor<?x?xi8, #scales_shuffle_encoding>
        util.return %4 : tensor<?x?xi8, #scales_shuffle_encoding>
    }}
{{% endraw %}}
    util.func private @{{{{kernel_name}}}}(%x: !x, %w: !w, %x_scale: !x_scale, %w_scale: !w_scale, %bias: !bias) -> !result {{
{{% if shuffle_scales %}}
        // Shuffle scales at runtime
        %x_scale_shuffle = util.call @shuffle_scales(%x_scale) : (!x_scale) -> tensor<?x?xi8, #scales_shuffle_encoding>
        %w_scale_shuffle = util.call @shuffle_scales(%w_scale) : (!w_scale) -> tensor<?x?xi8, #scales_shuffle_encoding>
        %result = util.call @asm_mxfp4_gemm(%x, %w, %x_scale_shuffle, %w_scale_shuffle, %bias) : (!x, !w, tensor<?x?xi8, #scales_shuffle_encoding>, tensor<?x?xi8, #scales_shuffle_encoding>, !bias) -> !result
{{% else %}}
        // Scales already shuffled offline
        %result = util.call @asm_mxfp4_gemm(%x, %w, %x_scale, %w_scale, %bias) : (!x, !w, !x_scale, !w_scale, !bias) -> !result
{{% endif %}}
        util.return %result : !result
    }}
}}
        """

    return MLIRSpec(mlir)


@mlir_kernel(
    inputs=(
        MLIRTensor[M, HALF_K, U8],
        MLIRTensor[N, HALF_K, U8],
        MLIRTensor[M, K_OVER_THIRTYTWO, U8],
        MLIRTensor[N, K_OVER_THIRTYTWO, U8],
        MLIRTensor[M_PADDED, N, F32],
    ),
    results=(MLIRTensor[M, N, F16],),
)
def _asm_fp4_gemm_regular(x, w, x_scale, w_scale, bias, result=None):
    return _build_mlir_spec(bin_asm_fp4_gemm)


@mlir_kernel(
    inputs=(
        MLIRTensor[M, HALF_K, U8],
        MLIRTensor[N, HALF_K, U8],
        MLIRTensor[M, K_OVER_THIRTYTWO, U8],
        MLIRTensor[N, K_OVER_THIRTYTWO, U8],
        MLIRTensor[M_PADDED, N, F32],
    ),
    results=(MLIRTensor[M, N, F16],),
)
def _asm_fp4_gemm_preshuffle(
    x, w, x_scale, w_scale, bias, result=None, shuffle_scales=True
):
    return _build_mlir_spec(bin_asm_fp4_gemm_preshuffle, shuffle_scales=shuffle_scales)


def asm_fp4_gemm(x, w, x_scale, w_scale, bias, use_preshuffle=False):
    """
    A4W4 asm gemm kernel dispatcher.

    Args:
        x: [M, K/2] f4x2 tensor
        w: [N, K/2] f4x2 tensor
        x_scale: [M, K/32] e8m0 scale tensor
        w_scale: [N, K/32] e8m0 scale tensor
        bias: [M, N] f32 bias tensor
        use_preshuffle: Version info - False (no preshuffle), True (v1, weights only),
                       or "v2" (weights + scales preshuffled)

    Returns:
        [M, N] f16 tensor
    """
    if use_preshuffle:
        logger.debug(
            f"Using preshuffle kernel for FP4 GEMM operation, version: {use_preshuffle}"
        )
        shuffle_scales = use_preshuffle != "v2"
        return _asm_fp4_gemm_preshuffle(
            x, w, x_scale, w_scale, bias, shuffle_scales=shuffle_scales
        )
    else:
        return _asm_fp4_gemm_regular(x, w, x_scale, w_scale, bias)
