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

from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from . import common


@dataclass
class GPUMMASchedule:
    m_size: z3.ArithRef
    n_size: z3.ArithRef
    k_size: z3.ArithRef

    m_subgroup_counts: z3.ArithRef
    n_subgroup_counts: z3.ArithRef

    m_tile_size: z3.ArithRef
    n_tile_size: z3.ArithRef
    k_tile_size: z3.ArithRef


@dataclass
class MMASingleSubgroupLayout:
    outer: tuple[z3.ArithRef, z3.ArithRef]
    thread: tuple[z3.ArithRef, z3.ArithRef]
    tstrides: tuple[z3.ArithRef, z3.ArithRef]
    element: tuple[z3.ArithRef, z3.ArithRef]


def create_mma_layout(prefix: str) -> MMASingleSubgroupLayout:
    return MMASingleSubgroupLayout(
        outer=(z3.Int(f"{prefix}_outer_x"), z3.Int(f"{prefix}_outer_y")),
        thread=(z3.Int(f"{prefix}_thread_x"), z3.Int(f"{prefix}_thread_y")),
        tstrides=(z3.Int(f"{prefix}_tstrides_x"), z3.Int(f"{prefix}_tstrides_y")),
        element=(z3.Int(f"{prefix}_element_x"), z3.Int(f"{prefix}_element_y")),
    )


def match_layout(
    layout_a: MMASingleSubgroupLayout, layout_b: MMASingleSubgroupLayout
) -> z3.BoolRef:
    return z3.And(
        layout_a.element[0] == layout_b.element[0],
        layout_a.element[1] == layout_b.element[1],
        layout_a.thread[0] == layout_b.thread[0],
        layout_a.thread[1] == layout_b.thread[1],
        layout_a.tstrides[0] == layout_b.tstrides[0],
        layout_a.tstrides[1] == layout_b.tstrides[1],
    )


def get_mfma_intrinsic_constraints(
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    intrinsic_m: z3.ArithRef,
    intrinsic_n: z3.ArithRef,
    intrinsic_k: z3.ArithRef,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
    lhs_layout: MMASingleSubgroupLayout | None = None,
    rhs_layout: MMASingleSubgroupLayout | None = None,
    acc_layout: MMASingleSubgroupLayout | None = None,
) -> z3.BoolRef:
    compatible_intrinsics = common.get_compatible_mfma_intrinsics(
        lhs_type, rhs_type, res_type, mma_intrinsics
    )
    assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"

    def get_mma_attr(instrinsic):
        return (
            iree_gpu.MMAAttr.get(instrinsic)
            if isinstance(instrinsic, iree_gpu.MMAIntrinsic)
            else iree_gpu.VirtualMMAAttr.get(instrinsic)
        )

    def match_layout(symbolic_layout, mma_layout) -> list[z3.BoolRef]:
        return [
            symbolic_layout.outer[0] == mma_layout.outer[0],
            symbolic_layout.outer[1] == mma_layout.outer[1],
            symbolic_layout.thread[0] == mma_layout.thread[0],
            symbolic_layout.thread[1] == mma_layout.thread[1],
            symbolic_layout.tstrides[0] == mma_layout.tstrides[0],
            symbolic_layout.tstrides[1] == mma_layout.tstrides[1],
            symbolic_layout.element[0] == mma_layout.element[0],
            symbolic_layout.element[1] == mma_layout.element[1],
        ]

    constraints = []
    for instr in compatible_intrinsics:
        if isinstance(instr, iree_gpu.MMAIntrinsic):
            mma_intrinsic_attr = iree_gpu.MMAIntrinsicAttr.get(instr)
        elif isinstance(instr, iree_gpu.VirtualMMAIntrinsic):
            mma_intrinsic_attr = iree_gpu.VirtualMMAIntrinsicAttr.get(instr)
        else:
            raise TypeError(f"Unsupported intrinsic type: {type(instr)}")
        mma_attr = get_mma_attr(instr)
        m, n, k = mma_attr.mnk_shape

        base_constraints = [
            intrinsic_m == m,
            intrinsic_n == n,
            intrinsic_k == k,
        ]

        if lhs_layout:
            mma_layout = iree_gpu.get_single_subgroup_layout(mma_intrinsic_attr, 0)
            assert isinstance(mma_layout, iree_gpu.GPUMMASingleSubgroupLayout)
            base_constraints += match_layout(lhs_layout, mma_layout)

        if rhs_layout:
            mma_layout = iree_gpu.get_single_subgroup_layout(mma_intrinsic_attr, 1)
            assert isinstance(mma_layout, iree_gpu.GPUMMASingleSubgroupLayout)
            base_constraints += match_layout(rhs_layout, mma_layout)

        if acc_layout:
            mma_layout = iree_gpu.get_single_subgroup_layout(mma_intrinsic_attr, 2)
            assert isinstance(mma_layout, iree_gpu.GPUMMASingleSubgroupLayout)
            base_constraints += match_layout(acc_layout, mma_layout)

        constraints.append(z3.And(*base_constraints))

    return z3.Or(*constraints)


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
    gpu_target_info: iree_gpu.TargetInfo,
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
    # Use minimum subgroup size for consistency with IREE side.
    # https://github.com/iree-org/iree/blob/c37c680ae6e71f715bd7c540909155061bc44491/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td#L623-L632
    target_subgroup_size = min(gpu_target_info.subgroup_size_choices)
    constraints += [
        subgroup_size == target_subgroup_size,
        wg_threads <= gpu_target_info.max_thread_count_per_workgroup,
    ]
    constraints += [
        get_mfma_intrinsic_constraints(
            lhs_type,
            rhs_type,
            res_type,
            intrinsic_mn,
            intrinsic_mn,
            intrinsic_k,
            gpu_target_info.mma_intrinsics,
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

    # Tile sizes need to evenly divide the problem size to be valid.
    constraints += [m >= intrinsic_mn, m <= 512, m <= M, M % m == 0]
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
    constraints += [shared_memory <= gpu_target_info.max_workgroup_memory_bytes]

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
    gpu_target_info: iree_gpu.TargetInfo,
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
    target_subgroup_size = min(gpu_target_info.subgroup_size_choices)
    constraints += [
        subgroup_size == target_subgroup_size,
        wg_threads <= gpu_target_info.max_thread_count_per_workgroup,
    ]
    constraints += [
        get_mfma_intrinsic_constraints(
            lhs_type,
            rhs_type,
            res_type,
            intrinsic_mn,
            intrinsic_mn,
            intrinsic_k,
            gpu_target_info.mma_intrinsics,
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
    constraints += [
        shared_memory * intrinsic_k <= gpu_target_info.max_workgroup_memory_bytes
    ]

    return constraints


def is_valid_vector_distribute_mma_schedule(
    matmul: common.MatmulShapeType,
    schedule: GPUMMASchedule,
    subgroup_size: z3.ArithRef,
    transposed_lhs: bool,
    transposed_rhs: bool,
) -> list[z3.BoolRef]:
    wg_threads = subgroup_size * schedule.m_subgroup_counts * schedule.n_subgroup_counts

    # Check alignment between matmul shape and tiling layout.
    schedule_aligned = z3.And(
        matmul.m % (schedule.m_subgroup_counts * schedule.m_tile_size * schedule.m_size)
        == 0,
        matmul.n % (schedule.n_subgroup_counts * schedule.n_tile_size * schedule.n_size)
        == 0,
        matmul.k % (schedule.k_tile_size * schedule.k_size) == 0,
    )

    kMaxVectorLoadBitWidth = 128
    bitwidth = matmul.rhs_type.width
    elems_per_thread = kMaxVectorLoadBitWidth // bitwidth

    m_wg_size = schedule.m_size * schedule.m_tile_size * schedule.m_subgroup_counts
    n_wg_size = schedule.n_size * schedule.n_tile_size * schedule.n_subgroup_counts
    k_wg_size = schedule.k_size * schedule.k_tile_size

    inner_lhs_dim = m_wg_size if transposed_lhs else k_wg_size
    inner_rhs_dim = k_wg_size if transposed_rhs else n_wg_size

    lhs_div = inner_lhs_dim / elems_per_thread
    rhs_div = inner_rhs_dim / elems_per_thread

    lhs_distributable = z3.Or(lhs_div % wg_threads == 0, wg_threads % lhs_div == 0)
    rhs_distributable = z3.Or(rhs_div % wg_threads == 0, wg_threads % rhs_div == 0)

    return [schedule_aligned, lhs_distributable, rhs_distributable]


def calculate_schedule_input_operands_shared_memory_usage_in_bytes(
    schedule: GPUMMASchedule,
    lhs_type: ir.IntegerType | ir.FloatType,
    rhs_type: ir.IntegerType | ir.FloatType,
) -> int | z3.ArithRef:
    """
    Computes the shared memory usage (in bytes) for input operands
    (LHS and RHS) in the given MMA schedule.
    """
    tile_m = schedule.m_size * schedule.m_tile_size * schedule.m_subgroup_counts
    tile_n = schedule.n_size * schedule.n_tile_size * schedule.n_subgroup_counts
    tile_k = schedule.k_size * schedule.k_tile_size

    lhs_bits = lhs_type.width
    rhs_bits = rhs_type.width

    lhs_bytes = (tile_m * tile_k * lhs_bits) / 8
    rhs_bytes = (tile_n * tile_k * rhs_bits) / 8

    total_shared_memory = lhs_bytes + rhs_bytes
    return total_shared_memory


def generate_attention_vector_distribute_constraints(
    qk_matmul: common.MatmulShapeType,
    pv_matmul: common.MatmulShapeType,
    transposed_q: bool,
    transposed_k: bool,
    transposed_v: bool,
    tile_sizes: list[z3.ArithRef],
    num_subgroups: int,
    subgroup_size: z3.ArithRef,
    qk_intrinsic_size: list[z3.ArithRef],
    pv_intrinsic_size: list[z3.ArithRef],
    subgroup_m_count: z3.ArithRef,
    subgroup_n_count: z3.ArithRef,
    can_reuse_qk_output_for_pv_input: z3.BoolRef,
    gpu_target_info: iree_gpu.TargetInfo,
):
    m_tile, n_tile, k_tile = tile_sizes
    qk_intrinsic_mn, qk_intrinsic_k = qk_intrinsic_size
    pv_intrinsic_mn, pv_intrinsic_k = pv_intrinsic_size

    wg_threads = z3.Int("wg_threads")

    qk_mma_acc_layout = create_mma_layout("qk_acc")
    pv_mma_lhs_layout = create_mma_layout("pv_lhs")
    pv_mma_rhs_layout = create_mma_layout("pv_rhs")
    pv_mma_acc_layout = create_mma_layout("pv_acc")

    constraints = []
    constraints += [
        get_mfma_intrinsic_constraints(
            lhs_type=common.ShapedType([qk_matmul.m, qk_matmul.k], qk_matmul.lhs_type),
            rhs_type=common.ShapedType([qk_matmul.k, qk_matmul.n], qk_matmul.rhs_type),
            res_type=common.ShapedType([qk_matmul.m, qk_matmul.n], qk_matmul.acc_type),
            intrinsic_m=qk_intrinsic_mn,
            intrinsic_n=qk_intrinsic_mn,
            intrinsic_k=qk_intrinsic_k,
            mma_intrinsics=gpu_target_info.mma_intrinsics,
            lhs_layout=None,
            rhs_layout=None,
            acc_layout=qk_mma_acc_layout,
        )
    ]

    constraints += [
        get_mfma_intrinsic_constraints(
            lhs_type=common.ShapedType([pv_matmul.m, pv_matmul.k], pv_matmul.lhs_type),
            rhs_type=common.ShapedType([pv_matmul.k, pv_matmul.n], pv_matmul.rhs_type),
            res_type=common.ShapedType([pv_matmul.m, pv_matmul.n], pv_matmul.acc_type),
            intrinsic_m=pv_intrinsic_mn,
            intrinsic_n=pv_intrinsic_mn,
            intrinsic_k=pv_intrinsic_k,
            mma_intrinsics=gpu_target_info.mma_intrinsics,
            lhs_layout=pv_mma_lhs_layout,
            rhs_layout=pv_mma_rhs_layout,
            acc_layout=pv_mma_acc_layout,
        )
    ]

    constraints += [match_layout(qk_mma_acc_layout, pv_mma_acc_layout)]

    constraints += [
        qk_matmul.m % qk_intrinsic_mn == 0,
        qk_matmul.n % qk_intrinsic_mn == 0,
        qk_matmul.k % qk_intrinsic_k == 0,
    ]
    constraints += [
        pv_matmul.m % pv_intrinsic_mn == 0,
        pv_matmul.n % pv_intrinsic_mn == 0,
        pv_matmul.k % pv_intrinsic_k == 0,
    ]

    constraints += [subgroup_m_count >= 1, subgroup_m_count <= 32]
    constraints += [subgroup_n_count == 1]

    subgroup_m_tile_count = z3.Int("sg_m_tcnt")
    subgroup_n_tile_count = z3.Int("sg_n_tcnt")
    subgroup_k_tile_count = z3.Int("sg_k_tcnt")

    can_reuse_lhs = match_layout(qk_mma_acc_layout, pv_mma_lhs_layout)
    can_reuse_rhs = match_layout(qk_mma_acc_layout, pv_mma_rhs_layout)
    constraints += [
        can_reuse_qk_output_for_pv_input == z3.Or(can_reuse_lhs, can_reuse_rhs)
    ]

    wg_threads = z3.Int("wg_threads")
    target_subgroup_size = min(gpu_target_info.subgroup_size_choices)
    constraints += [
        subgroup_size == target_subgroup_size,
        wg_threads <= gpu_target_info.max_thread_count_per_workgroup,
    ]
    constraints += [
        m_tile >= pv_intrinsic_mn,
        n_tile >= pv_intrinsic_mn,
        k_tile >= pv_intrinsic_k,
    ]
    constraints += [
        m_tile == subgroup_m_count * subgroup_m_tile_count * pv_intrinsic_mn
    ]
    constraints += [
        n_tile == subgroup_n_count * subgroup_n_tile_count * pv_intrinsic_mn
    ]
    constraints += [k_tile == subgroup_k_tile_count * pv_intrinsic_k]

    constraints += [n_tile <= 512, k_tile <= 512, m_tile <= 512]

    constraints += [wg_threads == subgroup_m_count * subgroup_n_count * subgroup_size]
    subgroups = subgroup_m_count * subgroup_n_count
    if num_subgroups > 0:
        constraints += [subgroups == num_subgroups]
    else:
        constraints += [subgroups >= 1, subgroups <= 10]

    pv_schedule = GPUMMASchedule(
        m_size=pv_intrinsic_mn,
        n_size=pv_intrinsic_mn,
        k_size=pv_intrinsic_k,
        m_subgroup_counts=subgroup_m_count,
        n_subgroup_counts=subgroup_n_count,
        m_tile_size=subgroup_m_tile_count,
        n_tile_size=subgroup_n_tile_count,
        k_tile_size=subgroup_k_tile_count,
    )

    qk_schedule = GPUMMASchedule(
        m_size=pv_intrinsic_mn,
        n_size=pv_intrinsic_k,
        k_size=qk_intrinsic_k,
        m_subgroup_counts=subgroup_m_count,
        n_subgroup_counts=1,
        m_tile_size=subgroup_m_tile_count,
        n_tile_size=subgroup_k_tile_count,
        k_tile_size=qk_matmul.k / qk_intrinsic_k,
    )

    constraints += is_valid_vector_distribute_mma_schedule(
        matmul=qk_matmul,
        schedule=qk_schedule,
        subgroup_size=subgroup_size,
        transposed_lhs=transposed_q,
        transposed_rhs=transposed_k,
    )

    constraints += is_valid_vector_distribute_mma_schedule(
        matmul=pv_matmul,
        schedule=pv_schedule,
        subgroup_size=subgroup_size,
        transposed_lhs=False,
        transposed_rhs=transposed_v,
    )

    qk_shared = calculate_schedule_input_operands_shared_memory_usage_in_bytes(
        qk_schedule, qk_matmul.lhs_type, qk_matmul.rhs_type
    )
    pv_shared = calculate_schedule_input_operands_shared_memory_usage_in_bytes(
        pv_schedule, pv_matmul.lhs_type, pv_matmul.rhs_type
    )

    # If QK output is reused for PV, only one PV operand is allocated; LHS and RHS are equal size.
    shared_memory = qk_shared + z3.If(
        can_reuse_qk_output_for_pv_input, pv_shared / 2, pv_shared
    )

    constraints += [shared_memory <= gpu_target_info.max_workgroup_memory_bytes]

    return constraints


def getMMAAttr(
    output_type: ir.IntegerType | ir.FloatType,
    m: int,
    n: int,
    k: int,
    lhs_type: ir.IntegerType | ir.FloatType,
    rhs_type: ir.IntegerType | ir.FloatType,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
) -> iree_gpu.MMAAttr | iree_gpu.VirtualMMAAttr:
    for mma_intrinsic in mma_intrinsics:
        if isinstance(mma_intrinsic, iree_gpu.MMAIntrinsic):
            mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
        else:
            mma_attr = iree_gpu.VirtualMMAAttr.get(mma_intrinsic)

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

    # If no matching intrinsic is found, raise an exception.
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
    mma_attr: iree_gpu.MMAAttr | iree_gpu.VirtualMMAAttr | None,
    workgroup_tile_sizes: list[int],
    reduction_tile_sizes: list[int],
    subgroup_tile_sizes: list[int],
    workgroup_sizes: tuple[int, int, int],
    subgroup_size: int,
    subgroup_basis_counts: list[int],
    subgroup_basis_mapping: list[int],
    promote_operands: list[int],
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
    pipeline_options_search_space: PipelineOptionsSearchSpace,
    allowed_waves_per_eu: list[int],
    padding: Optional[list[int]] = None,
) -> list[iree_codegen.CompilationInfoAttr]:
    subgroup_basis = [subgroup_basis_counts, subgroup_basis_mapping]
    # Create the LoweringConfigAttr.
    lowering_config_args = {
        "workgroup": workgroup_tile_sizes,
        "reduction": reduction_tile_sizes,
        "subgroup_basis": subgroup_basis,
        "promote_operands": promote_operands,
    }

    if mma_attr is not None:
        lowering_config_args["mma_kind"] = mma_attr

    if padding is not None:
        lowering_config_args["padding"] = padding

    if codegen_pipeline == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
        lowering_config_args["subgroup"] = subgroup_tile_sizes

    lowering_config = common.get_lowering_config(tuner_ctx, **lowering_config_args)

    # Create the TranslationInfoAttr.
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
