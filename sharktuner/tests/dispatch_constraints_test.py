# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest dispatch_constraints_test.py
"""

import pytest
import z3  # type: ignore

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu, iree_codegen  # type: ignore

from sharktuner import common
from sharktuner import dispatch_constraints

from sharktuner.test_utils import tuner_ctx


def test_generate_solutions(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.ContractionSizes([2048], [3840], [1280])
    contraction_dims = common.ContractionDimensions([0], [1], [2])
    lhs_type = common.ShapedType([2048, 1280], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([3840, 1280], tuner_ctx.type.f16)
    res_type = common.ShapedType([2048, 3840], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    configs = dispatch_constraints.generate_solutions(
        tuner_ctx,
        problem_size,
        4,
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )

    assert configs is not None


def test_generate_solutions_tile_and_fuse(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.ContractionSizes([5369], [112], [112])
    contraction_dims = common.ContractionDimensions([0], [1], [2])
    lhs_type = common.ShapedType([5369, 112], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([112, 112], tuner_ctx.type.f16)
    res_type = common.ShapedType([5369, 112], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )

    mma_intrinsics = [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
    ]

    solutions = list(
        dispatch_constraints.generate_solutions(
            tuner_ctx=tuner_ctx,
            problem_size=problem_size,
            num_subgrups=4,
            mma_intrinsics=mma_intrinsics,
            allowed_waves_per_eu=[2],
            pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        )
    )

    assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
    assert all(isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions)
    assert all(
        "padding =" in str(sol.lowering_config) for sol in solutions
    ), "Not all lowering configs have padding option"
    assert all(
        [int(x) for x in sol.lowering_config.attributes["promote_operands"]]
        == [0, 1, 2]
        for sol in solutions
    ), "Not all lowering configs have promote_operands = [0, 1, 2]"


def test_calculate_shared_memory_usage_in_bytes(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.ContractionSizes([1024], [1024], [1024])
    contraction_dims = common.ContractionDimensions([0], [1], [2])
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            problem_size, [512], [64], [128]
        )
        == 147456
    )

    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.i8)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            problem_size, [512], [64], [128]
        )
        == 81920
    )

    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.i32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            problem_size, [128], [64], [32]
        )
        == 12288
    )

    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            problem_size, [2, 64], [4, 16], [8, 4]
        )
        == 12288
    )


def test_adjust_problem_size_for_pipeline(
    tuner_ctx: common.TunerContext,
):
    # Test Matmul TileAndFuse. Expect no change.
    matmul_size = common.ContractionSizes(
        M=[32],
        N=[64],
        K=[128],
        B=[2],
    )
    contraction_dims = common.ContractionDimensions(
        m=[1],
        n=[2],
        k=[3],
        batch=[0],
    )
    lhs_type = common.ShapedType([2, 32, 128], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([2, 64, 128], tuner_ctx.type.f16)
    res_type = common.ShapedType([2, 32, 64], tuner_ctx.type.f32)
    matmul_problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    pipeline_options_space = dispatch_constraints.PipelineOptionsSearchSpace(
        prefetch_shared_memory=[True],
        no_reduce_shared_memory_bank_conflicts=[True, False],
        use_igemm_convolution=[None],
    )
    taf_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
    dispatch_constraints.adjust_problem_size_for_pipeline(
        problem_size=matmul_problem_size,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=taf_pipeline,
    )
    assert pipeline_options_space.prefetch_shared_memory == [True]
    assert pipeline_options_space.no_reduce_shared_memory_bank_conflicts == [
        True,
        False,
    ]
    assert pipeline_options_space.use_igemm_convolution == [None]
    assert matmul_problem_size.matmul_size.M == [32]
    assert matmul_problem_size.matmul_size.N == [64]
    assert matmul_problem_size.matmul_size.K == [128]
    assert matmul_problem_size.matmul_size.B == [2]
    assert matmul_problem_size.contraction_dims.m == [1]
    assert matmul_problem_size.contraction_dims.n == [2]
    assert matmul_problem_size.contraction_dims.k == [3]
    assert matmul_problem_size.contraction_dims.batch == [0]

    # Test Conv VectorDistribute. Expect no change.
    conv_size = common.ContractionSizes(
        M=[2, 32, 32],
        N=[256],
        K=[3, 3, 512],
    )
    contraction_dims = common.ContractionDimensions(
        m=[0, 1, 2],
        n=[3],
        k=[4, 5, 6],
    )
    lhs_type = common.ShapedType([2, 34, 34, 512], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([3, 3, 512, 256], tuner_ctx.type.f16)
    res_type = common.ShapedType([2, 32, 32, 256], tuner_ctx.type.f32)
    conv_problem_size = common.ProblemSize(
        conv_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.conv,
        contraction_dims,
    )
    vec_dist_pipeline = (
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    dispatch_constraints.adjust_problem_size_for_pipeline(
        problem_size=conv_problem_size,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=vec_dist_pipeline,
    )
    assert pipeline_options_space.prefetch_shared_memory == [True]
    assert pipeline_options_space.no_reduce_shared_memory_bank_conflicts == [
        True,
        False,
    ]
    assert pipeline_options_space.use_igemm_convolution == [None]
    assert conv_problem_size.matmul_size.M == [2, 32, 32]
    assert conv_problem_size.matmul_size.N == [256]
    assert conv_problem_size.matmul_size.K == [3, 3, 512]
    assert conv_problem_size.matmul_size.B == []
    assert conv_problem_size.contraction_dims.m == [0, 1, 2]
    assert conv_problem_size.contraction_dims.n == [3]
    assert conv_problem_size.contraction_dims.k == [4, 5, 6]
    assert conv_problem_size.contraction_dims.batch == []

    # Test Conv TileAndFuse. Expect flat K dims and use_igemm_convolution True.
    dispatch_constraints.adjust_problem_size_for_pipeline(
        problem_size=conv_problem_size,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=taf_pipeline,
    )
    assert pipeline_options_space.prefetch_shared_memory == [True]
    assert pipeline_options_space.no_reduce_shared_memory_bank_conflicts == [
        True,
        False,
    ]
    assert pipeline_options_space.use_igemm_convolution == [True]
    assert conv_problem_size.matmul_size.M == [2, 32, 32]
    assert conv_problem_size.matmul_size.N == [256]
    assert conv_problem_size.matmul_size.K == [4608]
    assert conv_problem_size.matmul_size.B == []
    assert conv_problem_size.contraction_dims.m == [0, 1, 2]
    assert conv_problem_size.contraction_dims.n == [3]
    assert conv_problem_size.contraction_dims.k == [4]
    assert conv_problem_size.contraction_dims.batch == []


def test_generate_tile_and_fuse_constraints_valid_input(
    tuner_ctx: common.TunerContext,
) -> None:
    matmul_size = common.ContractionSizes(
        M=[32],
        N=[64],
        K=[128],
        B=[2],
    )
    contraction_dims = common.ContractionDimensions(
        m=[1],
        n=[2],
        k=[3],
        batch=[0],
    )
    lhs_type = common.ShapedType([2, 32, 128], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([2, 64, 128], tuner_ctx.type.f16)
    res_type = common.ShapedType([2, 32, 64], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    # Define input parameters as z3 Ints
    m, n, k = (
        [z3.Int("m0")],
        [z3.Int("n0")],
        [z3.Int("k0")],
    )
    subgroup_m, subgroup_n = (
        [z3.Int("subgroup_m0")],
        [z3.Int("subgroup_n0")],
    )
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        z3.Int("wg_x"),
        z3.Int("wg_y"),
        z3.Int("wg_z"),
    )
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    constraints = dispatch_constraints.generate_tile_and_fuse_constraints(
        problem_size,
        [m, n, k, subgroup_m, subgroup_n],
        4,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable
    assert solver.check() == z3.sat


def test_generate_tile_and_fuse_constraints_invalid_input(
    tuner_ctx: common.TunerContext,
) -> None:
    # Define input parameters that should lead to unsatisfiable constraints
    matmul_size = common.ContractionSizes(
        M=[32],
        N=[64],
        K=[128],
        B=[2],
    )
    contraction_dims = common.ContractionDimensions(
        m=[1],
        n=[2],
        k=[3],
        batch=[0],
    )
    lhs_type = common.ShapedType([2, 32, 128], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([2, 64, 128], tuner_ctx.type.f16)
    res_type = common.ShapedType([2, 32, 64], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    # Define input parameters as z3 Ints
    m, n, k = (
        [z3.Int("m0")],
        [z3.Int("n0")],
        [z3.Int("k0")],
    )
    subgroup_m, subgroup_n = (
        [z3.Int("subgroup_m0")],
        [z3.Int("subgroup_n0")],
    )
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        z3.Int("wg_x"),
        z3.Int("wg_y"),
        z3.Int("wg_z"),
    )
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    constraints = dispatch_constraints.generate_tile_and_fuse_constraints(
        problem_size,
        [m, n, k, subgroup_m, subgroup_n],
        4,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )
    constraints.append(m[0] > 1000)  # Adding an additional unsatisfiable constraint

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == z3.unsat


def test_generate_vector_distribute_constraints_valid_input(
    tuner_ctx: common.TunerContext,
) -> None:
    matmul_size = common.ContractionSizes([1024], [1024], [1024])
    contraction_dims = common.ContractionDimensions([0], [1], [2])
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    # Define input parameters as z3 Ints
    m, n, k = (
        [z3.Int("m")],
        [z3.Int("n")],
        [z3.Int("k")],
    )
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        z3.Int("wg_x"),
        z3.Int("wg_y"),
        z3.Int("wg_z"),
    )
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    constraints = dispatch_constraints.generate_vector_distribute_constraints(
        problem_size,
        [m, n, k],
        4,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable
    assert solver.check() == z3.sat


def test_generate_vector_distribute_constraints_invalid_input(
    tuner_ctx: common.TunerContext,
) -> None:
    # Define input parameters that should lead to unsatisfiable constraints
    matmul_size = common.ContractionSizes([1024], [1024], [1024])
    contraction_dims = common.ContractionDimensions([0], [1], [2])
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    problem_size = common.ProblemSize(
        matmul_size,
        lhs_type,
        rhs_type,
        res_type,
        common.DispatchKind.contraction,
        contraction_dims,
    )
    m, n, k = (
        [z3.Int("m")],
        [z3.Int("n")],
        [z3.Int("k")],
    )
    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        z3.Int("wg_x"),
        z3.Int("wg_y"),
        z3.Int("wg_z"),
    )
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    constraints = dispatch_constraints.generate_vector_distribute_constraints(
        problem_size,
        [m, n, k],
        4,
        subgroup_size,
        [intrinsic_mn, intrinsic_k],
        [wg_x, wg_y, wg_z],
        sg_m_cnt,
        sg_n_cnt,
        [
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )
    constraints.append(m[0] > 1000)  # Adding an additional unsatisfiable constraint

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == z3.unsat
