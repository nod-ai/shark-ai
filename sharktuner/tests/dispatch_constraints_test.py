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

from iree.compiler.dialects import iree_gpu  # type: ignore

from sharktuner import common
from sharktuner import dispatch_constraints

from sharktuner.test_utils import tuner_ctx


def test_calculate_shared_memory_usage_in_bytes(tuner_ctx: common.TunerContext) -> None:
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            rhs_type, rhs_type, [512], [64], [128]
        )
        == 147456
    )

    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.i8)
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            lhs_type, rhs_type, [512], [64], [128]
        )
        == 81920
    )

    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.i32)
    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            lhs_type, rhs_type, [128], [64], [32]
        )
        == 12288
    )

    assert (
        dispatch_constraints.calculate_shared_memory_usage_in_bytes(
            lhs_type, rhs_type, [2, 64], [4, 16], [8, 4]
        )
        == 12288
    )


def test_generate_tile_and_fuse_constraints_valid_input(
    tuner_ctx: common.TunerContext,
) -> None:
    matmul_size = common.ContractionSizes(
        M=[32],
        N=[64],
        K=[128],
        B=[2],
    )
    lhs_type = common.ShapedType([2, 32, 128], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([2, 64, 128], tuner_ctx.type.f16)
    res_type = common.ShapedType([2, 32, 64], tuner_ctx.type.f32)

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
    tile_sizes = [m, n, k, subgroup_m, subgroup_n]

    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = (
        z3.Int("wg_x"),
        z3.Int("wg_y"),
        z3.Int("wg_z"),
    )
    wg_size = [wg_x, wg_y, wg_z]

    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    constraints = dispatch_constraints.generate_tile_and_fuse_constraints(
        matmul_size=matmul_size,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        res_type=res_type,
        tile_sizes=tile_sizes,
        num_subgroups=4,
        subgroup_size=subgroup_size,
        intrinsic_size=[intrinsic_mn, intrinsic_k],
        workgroup_size=wg_size,
        subgroup_m_count=sg_m_cnt,
        subgroup_n_count=sg_n_cnt,
        mma_intrinsics=[
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
    tile_sizes = [m, n, k, subgroup_m, subgroup_n]

    constraints = dispatch_constraints.generate_tile_and_fuse_constraints(
        matmul_size=matmul_size,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        res_type=res_type,
        tile_sizes=tile_sizes,
        num_subgroups=4,
        subgroup_size=subgroup_size,
        intrinsic_size=[intrinsic_mn, intrinsic_k],
        workgroup_size=[wg_x, wg_y, wg_z],
        subgroup_m_count=sg_m_cnt,
        subgroup_n_count=sg_n_cnt,
        mma_intrinsics=[
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
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    dispatch_kind = common.DispatchKind.contraction

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
        matmul_size=matmul_size,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        res_type=res_type,
        tile_sizes=[m, n, k],
        num_subgroups=4,
        subgroup_size=subgroup_size,
        intrinsic_size=[intrinsic_mn, intrinsic_k],
        workgroup_size=[wg_x, wg_y, wg_z],
        subgroup_m_count=sg_m_cnt,
        subgroup_n_count=sg_n_cnt,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
        dispatch_kind=dispatch_kind,
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
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    dispatch_kind = common.DispatchKind.contraction

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
        matmul_size=matmul_size,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        res_type=res_type,
        tile_sizes=[m, n, k],
        num_subgroups=4,
        subgroup_size=subgroup_size,
        intrinsic_size=[intrinsic_mn, intrinsic_k],
        workgroup_size=[wg_x, wg_y, wg_z],
        subgroup_m_count=sg_m_cnt,
        subgroup_n_count=sg_n_cnt,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
        dispatch_kind=dispatch_kind,
    )
    constraints.append(m[0] > 1000)  # Adding an additional unsatisfiable constraint

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable
    assert solver.check() == z3.unsat
