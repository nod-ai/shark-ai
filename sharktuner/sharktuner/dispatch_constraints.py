# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import math
import z3  # type: ignore
from typing import Optional
from dataclasses import dataclass, field

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_gpu  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore

from . import common


def get_mfma_intrinsic_constraints(
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    intrinsic_m: z3.ArithRef,
    intrinsic_n: z3.ArithRef,
    intrinsic_k: z3.ArithRef,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
) -> z3.BoolRef:
    compatible_intrinsics = common.get_compatible_mfma_intrinsics(
        lhs_type, rhs_type, res_type, mma_intrinsics
    )
    assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"

    mma_attrs = [iree_gpu.MMAAttr.get(mfma) for mfma in compatible_intrinsics]
    mnk_shapes = [mma_attr.mnk_shape for mma_attr in mma_attrs]

    return z3.Or(
        *(
            z3.And(
                intrinsic_m == m,
                intrinsic_n == n,
                intrinsic_k == k,
            )
            for m, n, k in mnk_shapes
        )
    )


def get_dispatch_constraints(
    matmul_size: common.ContractionSizes,
    dispatch_kind: common.DispatchKind,
    tile_m: z3.ArithRef,
    tile_n: z3.ArithRef,
    tile_k: z3.ArithRef,
) -> list[z3.BoolRef]:
    if dispatch_kind != common.DispatchKind.conv:
        return []

    max_tile_m = matmul_size.M[-1]
    [max_tile_n] = matmul_size.N
    max_tile_k = matmul_size.K[-1]
    conv_constraints = [
        # WARNING: This sometimes makes the constraints UNSAT for some reason.
        tile_m <= max_tile_m,
        tile_n <= max_tile_n,
        tile_k <= max_tile_k,
    ]
    return conv_constraints


def calculate_shared_memory_usage_in_bytes(
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    m: list[int] | list[z3.ArithRef],
    n: list[int] | list[z3.ArithRef],
    k: list[int] | list[z3.ArithRef],
) -> int | z3.ArithRef:
    lhs_memory = lhs_type.bitwidth // 8
    for size in m + k:
        lhs_memory *= size
    rhs_memory = rhs_type.bitwidth // 8
    for size in n + k:
        rhs_memory *= size
    return lhs_memory + rhs_memory


def generate_vector_distribute_constraints(
    matmul_size: common.ContractionSizes,
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    tile_sizes: list[list[z3.ArithRef]],
    num_subgroups: int,
    subgroup_size: z3.ArithRef,
    intrinsic_size: list[z3.ArithRef],
    workgroup_size: list[z3.ArithRef],
    subgroup_m_count: z3.ArithRef,
    subgroup_n_count: z3.ArithRef,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
    dispatch_kind: common.DispatchKind,
):
    M, N, K = (
        matmul_size.M[-1],
        matmul_size.N[-1],
        matmul_size.K[-1],
    )
    m_vars, n_vars, k_vars = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    wg_x, wg_y, wg_z = workgroup_size
    wg_threads = z3.Int("wg_threads")
    constraints = [wg_threads == wg_x * wg_y * wg_z]
    constraints += [subgroup_size == 64, wg_threads <= 1024]
    constraints += [
        get_mfma_intrinsic_constraints(
            lhs_type,
            rhs_type,
            res_type,
            intrinsic_mn,
            intrinsic_mn,
            intrinsic_k,
            mma_intrinsics,
        )
    ]
    subgroup_k_count = 1
    m = m_vars[-1]
    n = n_vars[-1]
    k = k_vars[-1]
    constraints += [v == 1 for v in m_vars[:-1] + n_vars[:-1] + k_vars[:-1]]
    constraints += [
        m >= intrinsic_mn,
        m <= 512,
        m <= M,
    ]
    constraints += [n >= intrinsic_mn, n <= 512, n <= N, N % n == 0]
    constraints += [k >= intrinsic_k, k <= 512, k <= K, K % k == 0]
    for x in (subgroup_m_count, subgroup_n_count):
        constraints += [x >= 1, x <= 32]

    subgroup_m_tile_count = z3.Int("sg_m_tcnt")
    subgroup_n_tile_count = z3.Int("sg_n_tcnt")
    subgroup_k_tile_count = z3.Int("sg_k_tcnt")
    for x in (subgroup_m_tile_count, subgroup_n_tile_count, subgroup_k_tile_count):
        constraints += [x >= 1, x <= 32]

    constraints += [m == subgroup_m_count * subgroup_m_tile_count * intrinsic_mn]
    constraints += [n == subgroup_n_count * subgroup_n_tile_count * intrinsic_mn]
    constraints += [k == subgroup_k_count * subgroup_k_tile_count * intrinsic_k]
    constraints += [wg_x == subgroup_size * subgroup_n_count]
    constraints += [wg_y == subgroup_m_count]
    constraints += [wg_z == subgroup_k_count]
    constraints += [z3.Or(wg_x <= n, wg_x <= m)]
    constraints += [k % intrinsic_mn == 0]
    constraints += [(k * n) % wg_threads == 0]
    constraints += [(k * m) % wg_threads == 0]
    subgroups = subgroup_m_count * subgroup_n_count
    if num_subgroups > 0:
        constraints += [subgroups == num_subgroups]
    else:
        constraints += [subgroups >= 1, subgroups <= 10]

    shared_memory = calculate_shared_memory_usage_in_bytes(
        lhs_type, rhs_type, [m], [n], [k]
    )
    constraints += [shared_memory <= 65536]

    constraints += get_dispatch_constraints(matmul_size, dispatch_kind, m, n, k)

    return constraints


def generate_tile_and_fuse_constraints(
    matmul_size: common.ContractionSizes,
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    tile_sizes: list[list[z3.ArithRef]],
    num_subgroups: int,
    subgroup_size: z3.ArithRef,
    intrinsic_size: list[z3.ArithRef],
    workgroup_size: list[z3.ArithRef],
    subgroup_m_count: z3.ArithRef,
    subgroup_n_count: z3.ArithRef,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
):
    M, N, K = list(matmul_size.M), list(matmul_size.N), list(matmul_size.K)
    m_tiles, n_tiles, k_tiles, subgroup_m_tiles, subgroup_n_tiles = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    M[-1] = ((M[-1] + intrinsic_mn - 1) / intrinsic_mn) * intrinsic_mn
    N[-1] = ((N[-1] + intrinsic_mn - 1) / intrinsic_mn) * intrinsic_mn
    K[-1] = ((K[-1] + intrinsic_k - 1) / intrinsic_k) * intrinsic_k
    wg_x, wg_y, wg_z = workgroup_size
    wg_threads = wg_x
    constraints = [wg_y == 1, wg_z == 1]
    constraints += [subgroup_size == 64, wg_threads <= 1024]
    constraints += [
        get_mfma_intrinsic_constraints(
            lhs_type,
            rhs_type,
            res_type,
            intrinsic_mn,
            intrinsic_mn,
            intrinsic_k,
            mma_intrinsics,
        )
    ]

    constraints += [
        m_tiles[-1] >= intrinsic_mn,
        m_tiles[-1] % intrinsic_mn == 0,
        n_tiles[-1] >= intrinsic_mn,
        n_tiles[-1] % intrinsic_mn == 0,
        k_tiles[-1] * intrinsic_k <= K[-1],
        math.prod(m_tiles) <= 512,
        math.prod(n_tiles) <= 512,
        math.prod(k_tiles) <= 512 / intrinsic_k,
    ]
    constraints += [m_shape % m == 0 for m, m_shape in zip(m_tiles, M)]
    constraints += [n_shape % n == 0 for n, n_shape in zip(n_tiles, N)]
    constraints += [k_shape % k == 0 for k, k_shape in zip(k_tiles[:-1], K[:-1])]
    constraints += [m >= 1 for m in m_tiles]
    constraints += [n >= 1 for n in n_tiles]
    constraints += [k >= 1 for k in k_tiles]
    constraints += [K[-1] % (k_tiles[-1] * intrinsic_k) == 0]
    constraints += [m <= m_shape for m, m_shape in zip(m_tiles, M)]
    constraints += [n <= n_shape for n, n_shape in zip(n_tiles, N)]
    constraints += [k <= k_shape for k, k_shape in zip(k_tiles[:-1], K[:-1])]
    constraints += [(k_tiles[-1] * intrinsic_k) <= K[-1]]
    for x in (subgroup_m_count, subgroup_n_count):
        constraints += [x >= 1, x <= 32]

    constraints += [
        m % m_subgroup == 0
        for m, m_subgroup in zip(m_tiles[:-1], subgroup_m_tiles[:-1])
    ]
    constraints += [
        n % n_subgroup == 0
        for n, n_subgroup in zip(n_tiles[:-1], subgroup_n_tiles[:-1])
    ]
    constraints += [m_tiles[-1] % (subgroup_m_tiles[-1] * intrinsic_mn) == 0]
    constraints += [n_tiles[-1] % (subgroup_n_tiles[-1] * intrinsic_mn) == 0]
    constraints += [m_subgroup >= 1 for m_subgroup in subgroup_m_tiles]
    constraints += [n_subgroup >= 1 for n_subgroup in subgroup_n_tiles]

    constraints += [
        math.prod(m_tiles)
        == math.prod(subgroup_m_tiles) * subgroup_m_count * intrinsic_mn
    ]
    constraints += [
        math.prod(n_tiles)
        == math.prod(subgroup_n_tiles) * subgroup_n_count * intrinsic_mn
    ]
    subgroups = subgroup_m_count * subgroup_n_count
    if num_subgroups > 0:
        constraints += [subgroups == num_subgroups]
    else:
        constraints += [subgroups >= 1, subgroups <= 10]
    constraints += [wg_threads == subgroups * subgroup_size]

    shared_memory = calculate_shared_memory_usage_in_bytes(
        lhs_type, rhs_type, m_tiles, n_tiles, k_tiles
    )
    constraints += [shared_memory * intrinsic_k <= 65536]

    return constraints


def getMMAAttr(
    output_type: ir.IntegerType | ir.FloatType,
    m: int,
    n: int,
    k: int,
    lhs_type: ir.IntegerType | ir.FloatType,
    rhs_type: ir.IntegerType | ir.FloatType,
) -> iree_gpu.MMAAttr:
    for mma_intrinsic in iree_gpu.MMAIntrinsic:
        mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
        a_type, b_type, c_type = mma_attr.abc_element_types
        mnk = mma_attr.mnk_shape
        if (
            isinstance(a_type, type(lhs_type))
            and isinstance(b_type, type(rhs_type))
            and isinstance(c_type, type(output_type))
            and m == mnk[0]
            and n == mnk[1]
            and k == mnk[2]
        ):
            return mma_attr
        # If no matching intrinsic is found, raise an exception
    raise ValueError(
        f"No matching MMA intrinsic found for "
        f"output_type={output_type}, lhs_type={lhs_type}, rhs_type={rhs_type}, "
        f"m={m}, n={n}, k={k}."
    )


@dataclass
class PipelineOptionsSearchSpace:
    prefetch_shared_memory: list[Optional[bool]] = field(default_factory=lambda: [None])
    no_reduce_shared_memory_bank_conflicts: list[Optional[bool]] = field(
        default_factory=lambda: [None]
    )
    use_igemm_convolution: list[Optional[bool]] = field(default_factory=lambda: [None])


def generate_allowed_pipeline_options(
    pipeline_options_search_space: PipelineOptionsSearchSpace,
) -> list[iree_gpu.PipelineOptionsAttr]:
    pipeline_options_list = []
    for psm in pipeline_options_search_space.prefetch_shared_memory:
        for (
            nrbc
        ) in pipeline_options_search_space.no_reduce_shared_memory_bank_conflicts:
            for igemm in pipeline_options_search_space.use_igemm_convolution:
                pipeline_options_list.append(
                    iree_gpu.PipelineOptionsAttr.get(
                        prefetch_shared_memory=psm,
                        no_reduce_shared_memory_bank_conflicts=nrbc,
                        use_igemm_convolution=igemm,
                    )
                )
    return pipeline_options_list


def generate_compilation_infos(
    tuner_ctx: common.TunerContext,
    mma_attr: iree_gpu.MMAAttr,
    workgroup_tile_sizes: list[int],
    reduction_tile_sizes: list[int],
    subgroup_tile_sizes: list[int],
    workgroup_sizes: tuple[int, int, int],
    subgroup_size: int,
    subgroup_m_count: int,
    subgroup_n_count: int,
    promote_operands: list[int],
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
    pipeline_options_search_space: PipelineOptionsSearchSpace,
    allowed_waves_per_eu: list[int],
    padding: Optional[list[int]] = None,
) -> list[iree_codegen.CompilationInfoAttr]:
    # Create the LoweringConfigAttr.
    lowering_config_args = {
        "tuner_ctx": tuner_ctx,
        "mma_kind": mma_attr,
        "workgroup": workgroup_tile_sizes,
        "reduction": reduction_tile_sizes,
        "subgroup_m_count": subgroup_m_count,
        "subgroup_n_count": subgroup_n_count,
        "promote_operands": promote_operands,
    }

    if padding is not None:
        lowering_config_args["padding"] = padding

    if codegen_pipeline == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
        lowering_config_args["subgroup"] = subgroup_tile_sizes

    lowering_config = common.get_lowering_config(**lowering_config_args)

    # Create the TranslationInfoAttr
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(codegen_pipeline)
    pipeline_options_list = generate_allowed_pipeline_options(
        pipeline_options_search_space
    )
    wg_x, wg_y, wg_z = workgroup_sizes
    compilation_infos = []
    for pipeline_options in pipeline_options_list:
        for waves_per_eu in allowed_waves_per_eu:
            config_dict = common.get_translation_info_config(
                pipeline_options, waves_per_eu
            )
            translation_info = iree_codegen.TranslationInfoAttr.get(
                pipeline_attr,
                None,
                [wg_x, wg_y, wg_z],
                subgroup_size,
                config_dict,
            )
            compilation_infos.append(
                iree_codegen.CompilationInfoAttr.get(lowering_config, translation_info)
            )
    return compilation_infos
