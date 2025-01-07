# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import math
import z3  # type: ignore
from typing import Iterator

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_gpu  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore

from .common import *


def get_mfma_intrinsic_constraints(
    problem_size: ProblemSize,
    intrinsic_m: z3.ArithRef,
    intrinsic_n: z3.ArithRef,
    intrinsic_k: z3.ArithRef,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
) -> z3.BoolRef:
    compatible_intrinsics = get_compatible_mfma_intrinsics(problem_size, mma_intrinsics)
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
    problem_size: ProblemSize,
    tile_m: z3.ArithRef,
    tile_n: z3.ArithRef,
    tile_k: z3.ArithRef,
) -> list[z3.BoolRef]:
    if problem_size.dispatch_kind != DispatchKind.conv:
        return []

    dim_info = ConvDimInfo.from_problem_size(problem_size)
    conv_constraints = []
    # WARNING: This sometimes makes the constraints UNSAT for some reason.
    conv_constraints += [tile_m <= dim_info.ow]
    conv_constraints += [tile_n <= dim_info.oc]
    conv_constraints += [tile_k <= dim_info.ic]
    return conv_constraints


def calculate_shared_memory_usage_in_bytes(
    problem_size: ProblemSize,
    m: list[int] | list[z3.ArithRef],
    n: list[int] | list[z3.ArithRef],
    k: list[int] | list[z3.ArithRef],
) -> int | z3.ArithRef:
    lhs_memory = problem_size.lhs_type.bitwidth // 8
    for size in m + k:
        lhs_memory *= size
    rhs_memory = problem_size.rhs_type.bitwidth // 8
    for size in n + k:
        rhs_memory *= size
    return lhs_memory + rhs_memory


def generate_vector_distribute_constraints(
    problem_size: ProblemSize,
    tile_sizes: list[list[z3.ArithRef]],
    num_subgroups: int,
    subgroup_size: z3.ArithRef,
    intrinsic_size: list[z3.ArithRef],
    workgroup_size: list[z3.ArithRef],
    subgroup_m_count: z3.ArithRef,
    subgroup_n_count: z3.ArithRef,
    waves_per_eu: z3.ArithRef,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
):
    M, N, K = (
        problem_size.matmul_size.M[-1],
        problem_size.matmul_size.N[-1],
        problem_size.matmul_size.K[-1],
    )
    m_vars, n_vars, k_vars = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    wg_x, wg_y, wg_z = workgroup_size
    wg_threads = z3.Int("wg_threads")
    constraints = [wg_threads == wg_x * wg_y * wg_z]
    constraints += [subgroup_size == 64, wg_threads <= 1024]
    constraints += [
        get_mfma_intrinsic_constraints(
            problem_size, intrinsic_mn, intrinsic_mn, intrinsic_k, mma_intrinsics
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

    constraints += [waves_per_eu == 2]
    # constraints += [z3.Or(waves_per_eu == 2, waves_per_eu == 3, waves_per_eu == 4)]

    shared_memory = calculate_shared_memory_usage_in_bytes(problem_size, [m], [n], [k])
    constraints += [shared_memory <= 65536]

    constraints += get_dispatch_constraints(problem_size, m, n, k)

    return constraints


def generate_tile_and_fuse_constraints(
    problem_size: ProblemSize,
    tile_sizes: list[list[z3.ArithRef]],
    num_subgroups: int,
    subgroup_size: z3.ArithRef,
    intrinsic_size: list[z3.ArithRef],
    workgroup_size: list[z3.ArithRef],
    subgroup_m_count: z3.ArithRef,
    subgroup_n_count: z3.ArithRef,
    waves_per_eu: z3.ArithRef,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
):
    M, N, K = problem_size.MNK
    m_tiles, n_tiles, k_tiles, subgroup_m_tiles, subgroup_n_tiles = tile_sizes
    intrinsic_mn, intrinsic_k = intrinsic_size
    wg_x, wg_y, wg_z = workgroup_size
    wg_threads = z3.Int("wg_threads")
    constraints = [wg_x == wg_threads, wg_y == 1, wg_z == 1]
    constraints += [subgroup_size == 64, wg_threads <= 1024]
    constraints += [
        get_mfma_intrinsic_constraints(
            problem_size, intrinsic_mn, intrinsic_mn, intrinsic_k, mma_intrinsics
        )
    ]
    subgroup_k_count = 1

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
    constraints += [m >= 0 for m in m_tiles]
    constraints += [n >= 0 for n in n_tiles]
    constraints += [k >= 0 for k in k_tiles]
    constraints += [K[-1] % (k_tiles[-1] * intrinsic_k) == 0]
    constraints += [m <= m_shape for m, m_shape in zip(m_tiles, M)]
    constraints += [n <= n_shape for n, n_shape in zip(n_tiles, N)]
    constraints += [k <= k_shape for k, k_shape in zip(k_tiles[:-1], K[:-1])]
    constraints += [(k_tiles[-1] * intrinsic_k) <= K[-1]]
    for x in (subgroup_m_count, subgroup_n_count):
        constraints += [x >= 1, x <= 32]

    subgroup_m_tile_count = z3.Int("sg_m_tcnt")
    subgroup_n_tile_count = z3.Int("sg_n_tcnt")
    subgroup_k_tile_count = z3.Int("sg_k_tcnt")
    for x in (subgroup_m_tile_count, subgroup_n_tile_count, subgroup_k_tile_count):
        constraints += [x >= 1, x <= 32]
    constraints += [math.prod(subgroup_m_tiles) == subgroup_m_tile_count]
    constraints += [math.prod(subgroup_n_tiles) == subgroup_n_tile_count]
    constraints += [
        m % m_subgroup == 0 for m, m_subgroup in zip(m_tiles, subgroup_m_tiles)
    ]
    constraints += [
        n % n_subgroup == 0 for n, n_subgroup in zip(n_tiles, subgroup_n_tiles)
    ]
    constraints += [m_subgroup > 0 for m_subgroup in subgroup_m_tiles]
    constraints += [n_subgroup > 0 for n_subgroup in subgroup_n_tiles]

    constraints += [
        math.prod(m_tiles) == subgroup_m_count * subgroup_m_tile_count * intrinsic_mn
    ]
    constraints += [
        math.prod(n_tiles) == subgroup_n_count * subgroup_n_tile_count * intrinsic_mn
    ]
    constraints += [math.prod(k_tiles) == subgroup_k_count * subgroup_k_tile_count]
    subgroups = subgroup_m_count * subgroup_n_count
    if num_subgroups > 0:
        constraints += [subgroups == num_subgroups]
    else:
        constraints += [subgroups >= 1, subgroups <= 10]
    constraints += [wg_threads == subgroups * subgroup_size]

    constraints += [waves_per_eu == 2]

    shared_memory = calculate_shared_memory_usage_in_bytes(
        problem_size, m_tiles, n_tiles, k_tiles
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


def generate_solutions(
    tuner_ctx: TunerContext,
    problem_size: ProblemSize,
    num_subgrups: int,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
) -> Iterator[iree_codegen.CompilationInfoAttr]:
    M, N, K = problem_size.MNK
    tuner_ctx.logger.info(f"{M},{N},{K}")
    m_vars = [z3.Int(f"m{i}") for i in range(len(M))]
    n_vars = [z3.Int(f"n{i}") for i in range(len(N))]
    k_vars = [z3.Int(f"k{i}") for i in range(len(K))]
    subgroup_m_vars = [z3.Int(f"subgroup_m{i}") for i in range(len(M))]
    subgroup_n_vars = [z3.Int(f"subgroup_n{i}") for i in range(len(N))]
    # m, n, k = z3.Int("m"), z3.Int("n"), z3.Int("k")
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")
    waves_per_eu = z3.Int("waves_per_eu")
    all_vars = (
        m_vars
        + n_vars
        + k_vars
        + [
            subgroup_size,
            intrinsic_mn,
            intrinsic_k,
            wg_x,
            wg_y,
            wg_z,
            sg_m_cnt,
            sg_n_cnt,
            waves_per_eu,
        ]
    )

    solver = z3.Solver()
    match codegen_pipeline:
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute:
            constraints = generate_vector_distribute_constraints(
                problem_size,
                [m_vars, n_vars, k_vars],
                num_subgrups,
                subgroup_size,
                [intrinsic_mn, intrinsic_k],
                [wg_x, wg_y, wg_z],
                sg_m_cnt,
                sg_n_cnt,
                waves_per_eu,
                mma_intrinsics,
            )
            constraints += [v == 0 for v in subgroup_m_vars + subgroup_n_vars]
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
            constraints = generate_tile_and_fuse_constraints(
                problem_size,
                [m_vars, n_vars, k_vars, subgroup_m_vars, subgroup_n_vars],
                num_subgrups,
                subgroup_size,
                [intrinsic_mn, intrinsic_k],
                [wg_x, wg_y, wg_z],
                sg_m_cnt,
                sg_n_cnt,
                waves_per_eu,
                mma_intrinsics,
            )
    solver.add(z3.simplify(z3.And(constraints)))
    tuner_ctx.logger.debug(f"Initial constraints: {solver}")

    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()
        mma_attr = getMMAAttr(
            problem_size.res_type.element_type,
            lookup(intrinsic_mn),
            lookup(intrinsic_mn),
            lookup(intrinsic_k),
            problem_size.lhs_type.element_type,
            problem_size.rhs_type.element_type,
        )

        def set_cdim_tile_sizes(tile_sizes, contraction_dims, csizes):
            for dim, size in zip(contraction_dims, csizes):
                tile_sizes[dim] = size

        # Get workgroup tile sizes.
        workgroup_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(problem_size.contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            problem_size.contraction_dims.m,
            [lookup(v) for v in m_vars],
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            problem_size.contraction_dims.n,
            [lookup(v) for v in n_vars],
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            problem_size.contraction_dims.batch,
            [1] * len(problem_size.contraction_dims.batch),
        )

        # Get subgroup tile sizes.
        subgroup_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(problem_size.contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            problem_size.contraction_dims.m,
            [lookup(v) for v in subgroup_m_vars],
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            problem_size.contraction_dims.n,
            [lookup(v) for v in subgroup_n_vars],
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            problem_size.contraction_dims.batch,
            [1] * len(problem_size.contraction_dims.batch),
        )

        # Get reduction tile sizes.
        reduction_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(problem_size.contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            reduction_tile_sizes,
            problem_size.contraction_dims.k,
            [lookup(v) for v in k_vars],
        )

        # Create the LoweringConfigAttr.
        lowering_config_args = {
            "tuner_ctx": tuner_ctx,
            "mma_kind": mma_attr,
            "workgroup": workgroup_tile_sizes,
            "reduction": reduction_tile_sizes,
            "subgroup_m_count": lookup(sg_m_cnt),
            "subgroup_n_count": lookup(sg_n_cnt),
        }
        if (
            codegen_pipeline
            == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
        ):
            lowering_config_args["subgroup"] = subgroup_tile_sizes
        lowering_config = get_lowering_config(**lowering_config_args)

        # Create the TranslationInfoAttr
        pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
            codegen_pipeline
        )
        pipeline_options = iree_gpu.PipelineOptionsAttr.get()
        config_dict = get_translation_info_config(
            pipeline_options, lookup(waves_per_eu)
        )
        translation_info = iree_codegen.TranslationInfoAttr.get(
            pipeline_attr,
            None,
            [lookup(wg_x), lookup(wg_y), lookup(wg_z)],
            lookup(subgroup_size),
            config_dict,
        )

        # Create the CompilationInfoAttr.
        compilation_info = iree_codegen.CompilationInfoAttr.get(
            lowering_config, translation_info
        )

        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1
        yield compilation_info
