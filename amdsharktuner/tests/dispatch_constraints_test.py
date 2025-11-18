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

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import common, dispatch_constraints

from amdsharktuner.test_utils import tuner_ctx


@pytest.fixture
def gpu_target_info(tuner_ctx: common.TunerContext) -> iree_gpu.TargetInfo:
    context = tuner_ctx.mlir_ctx
    return iree_gpu.TargetInfo(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[64],
        max_workgroup_sizes=[1024, 1024, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        workgroup_count=304,
        simds_per_workgroup=4,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )


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
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
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

    # Define input parameters as z3 Ints.
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
        gpu_target_info=gpu_target_info,
    )

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable.
    assert solver.check() == z3.sat


def test_generate_tile_and_fuse_constraints_invalid_input(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    # Define input parameters that should lead to unsatisfiable constraints.
    matmul_size = common.ContractionSizes(
        M=[32],
        N=[64],
        K=[128],
        B=[2],
    )
    lhs_type = common.ShapedType([2, 32, 128], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([2, 64, 128], tuner_ctx.type.f16)
    res_type = common.ShapedType([2, 32, 64], tuner_ctx.type.f32)
    # Define input parameters as z3 Ints.
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
        gpu_target_info=gpu_target_info,
    )
    constraints.append(m[0] > 1000)  # Adding an additional unsatisfiable constraint.

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable.
    assert solver.check() == z3.unsat


def test_generate_vector_distribute_constraints_valid_input(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    matmul_size = common.ContractionSizes([1024], [1024], [1024])
    lhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([1024, 1024], tuner_ctx.type.f16)
    res_type = common.ShapedType([1024, 1024], tuner_ctx.type.f32)
    dispatch_kind = common.DispatchKind.contraction

    # Define input parameters as z3 Ints.
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
        gpu_target_info=gpu_target_info,
        dispatch_kind=dispatch_kind,
    )

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are satisfiable.
    assert solver.check() == z3.sat


def test_generate_vector_distribute_constraints_invalid_input(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    # Define input parameters that should lead to unsatisfiable constraints.
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
        gpu_target_info=gpu_target_info,
        dispatch_kind=dispatch_kind,
    )
    constraints.append(m[0] > 1000)  # Adding an additional unsatisfiable constraint

    solver = z3.Solver()
    solver.add(constraints)

    # Check if the constraints are unsatisfiable.
    assert solver.check() == z3.unsat


def test_is_valid_mma_schedule(
    tuner_ctx: common.TunerContext,
) -> None:
    matmul = common.MatmulShapeType(
        m=128,
        n=128,
        k=64,
        lhs_type=tuner_ctx.type.f16,
        rhs_type=tuner_ctx.type.f16,
        acc_type=tuner_ctx.type.f32,
    )

    subgroup_size = z3.Int("subgroup_size")
    schedule = dispatch_constraints.GPUMMASchedule(
        m_size=z3.IntVal(16),
        n_size=z3.IntVal(16),
        k_size=z3.IntVal(16),
        m_subgroup_counts=z3.IntVal(1),
        n_subgroup_counts=z3.IntVal(1),
        m_tile_size=z3.IntVal(2),
        n_tile_size=z3.IntVal(2),
        k_tile_size=z3.IntVal(2),
    )

    constraints = dispatch_constraints.is_valid_vector_distribute_mma_schedule(
        matmul=matmul,
        schedule=schedule,
        subgroup_size=subgroup_size,
        transposed_lhs=False,
        transposed_rhs=False,
    )

    solver = z3.Solver()
    solver.add(subgroup_size == 64)
    solver.add(constraints)

    assert solver.check() == z3.sat

    matmul = common.MatmulShapeType(
        m=130,
        n=128,
        k=64,
        lhs_type=tuner_ctx.type.f16,
        rhs_type=tuner_ctx.type.f16,
        acc_type=tuner_ctx.type.f32,
    )
    constraints = dispatch_constraints.is_valid_vector_distribute_mma_schedule(
        matmul=matmul,
        schedule=schedule,
        subgroup_size=subgroup_size,
        transposed_lhs=False,
        transposed_rhs=False,
    )

    solver = z3.Solver()
    solver.add(subgroup_size == 64)
    solver.add(constraints)

    assert solver.check() == z3.unsat


def test_get_mfma_intrinsic_constraints(
    tuner_ctx: common.TunerContext,
) -> None:
    lhs_type = common.ShapedType([16, 16], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([16, 16], tuner_ctx.type.f16)
    res_type = common.ShapedType([16, 16], tuner_ctx.type.f32)

    intrinsic_m = z3.Int("intrinsic_m")
    intrinsic_n = z3.Int("intrinsic_n")
    intrinsic_k = z3.Int("intrinsic_k")

    constraints = dispatch_constraints.get_mfma_intrinsic_constraints(
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        res_type=res_type,
        intrinsic_m=intrinsic_m,
        intrinsic_n=intrinsic_n,
        intrinsic_k=intrinsic_k,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_32x32x16_F8E4M3FNUZ,
        ],
    )

    solver = z3.Solver()
    solver.add(constraints)
    assert solver.check() == z3.sat

    model = solver.model()
    m_val = model[intrinsic_m].as_long()
    n_val = model[intrinsic_n].as_long()
    k_val = model[intrinsic_k].as_long()
    assert (m_val, n_val, k_val) in [(16, 16, 16), (32, 32, 8)]


def test_match_layout():
    layout_a = dispatch_constraints.MMASingleSubgroupLayout(
        outer=(z3.IntVal(0), z3.IntVal(0)),
        thread=(z3.IntVal(1), z3.IntVal(2)),
        tstrides=(z3.IntVal(2), z3.IntVal(4)),
        element=(z3.IntVal(8), z3.IntVal(16)),
    )

    layout_b = dispatch_constraints.MMASingleSubgroupLayout(
        outer=(z3.IntVal(0), z3.IntVal(0)),
        thread=(z3.IntVal(3), z3.IntVal(2)),
        tstrides=(z3.IntVal(2), z3.IntVal(4)),
        element=(z3.IntVal(8), z3.IntVal(16)),
    )

    solver = z3.Solver()
    solver.add(dispatch_constraints.match_layout(layout_a, layout_b))
    assert solver.check() == z3.unsat

    layout_b.thread = (z3.IntVal(1), z3.IntVal(2))
    solver = z3.Solver()
    solver.add(dispatch_constraints.match_layout(layout_a, layout_b))
    assert solver.check() == z3.sat


def test_generate_attention_vector_distribute_constraints(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    f32 = tuner_ctx.type.f32
    f16 = tuner_ctx.type.f16

    qk_matmul = common.MatmulShapeType(
        m=128, n=128, k=64, lhs_type=f16, rhs_type=f16, acc_type=f32
    )
    pv_matmul = common.MatmulShapeType(
        m=128, n=128, k=64, lhs_type=f16, rhs_type=f16, acc_type=f32
    )

    m_tile = z3.Int("m_tile")
    n_tile = z3.Int("n_tile")
    k_tile = z3.Int("k_tile")
    tile_sizes = [m_tile, n_tile, k_tile]

    subgroup_size = z3.IntVal(64)
    qk_intrinsic_size = [z3.IntVal(16), z3.IntVal(16)]
    pv_intrinsic_size = [z3.IntVal(16), z3.IntVal(16)]

    subgroup_m_count = z3.Int("subgroup_m_count")
    subgroup_n_count = z3.Int("subgroup_n_count")
    can_reuse_qk_output_for_pv_input = z3.Bool("can_reuse_qk_output_for_pv_input")

    constraints = dispatch_constraints.generate_attention_vector_distribute_constraints(
        qk_matmul=qk_matmul,
        pv_matmul=pv_matmul,
        transposed_q=False,
        transposed_k=False,
        transposed_v=False,
        tile_sizes=tile_sizes,
        num_subgroups=4,
        subgroup_size=subgroup_size,
        qk_intrinsic_size=qk_intrinsic_size,
        pv_intrinsic_size=pv_intrinsic_size,
        subgroup_m_count=subgroup_m_count,
        subgroup_n_count=subgroup_n_count,
        can_reuse_qk_output_for_pv_input=can_reuse_qk_output_for_pv_input,
        gpu_target_info=gpu_target_info,
    )

    solver = z3.Solver()
    solver.add(constraints)
    assert solver.check() == z3.sat

    # Add a conflicting constraint for invalid test.
    constraints.append(m_tile > 1024)
    solver = z3.Solver()
    solver.add(constraints)
    assert solver.check() == z3.unsat
