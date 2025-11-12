# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import pytest
from typing import Optional

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from sharktuner import candidate_ordering, common


@pytest.fixture
def sample_knobs() -> list[Optional[common.KnobAssignment]]:
    knob_1 = common.LLVMGPUVectorDistributeContractionKnobs(
        M=2048,
        N=10240,
        K=1280,
        tile_m=128,
        tile_n=64,
        tile_k=64,
        wg_x=64,
        wg_y=2,
        wg_z=1,
        subgroup_m_cnt=2,
        subgroup_n_cnt=1,
        intrinsic_mn=32,
        intrinsic_k=8,
        subgroup_m=0,
        subgroup_n=0,
        subgroup_k=0,
    )
    knob_2 = common.LLVMGPUVectorDistributeContractionKnobs(
        M=2048,
        N=10240,
        K=1280,
        tile_m=64,
        tile_n=320,
        tile_k=80,
        wg_x=320,
        wg_y=1,
        wg_z=1,
        subgroup_m_cnt=1,
        subgroup_n_cnt=5,
        intrinsic_mn=16,
        intrinsic_k=16,
        subgroup_m=0,
        subgroup_n=0,
        subgroup_k=0,
    )
    knob_3 = common.LLVMGPUVectorDistributeContractionKnobs(
        M=2048,
        N=10240,
        K=1280,
        tile_m=64,
        tile_n=256,
        tile_k=16,
        wg_x=256,
        wg_y=2,
        wg_z=1,
        subgroup_m_cnt=2,
        subgroup_n_cnt=4,
        intrinsic_mn=16,
        intrinsic_k=16,
        subgroup_m=0,
        subgroup_n=0,
        subgroup_k=0,
    )
    return [knob_1, knob_2, knob_3]


@pytest.fixture
def target_info() -> iree_gpu.TargetInfo:
    context = ir.Context()

    return iree_gpu.TargetInfo(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[32, 64],
        max_workgroup_sizes=[256, 512, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        workgroup_count=304,
        simds_per_workgroup=4,
        mma_intrinsics=[],
    )


def test_math_expression() -> None:
    assert candidate_ordering.is_pow2(1) == True
    assert candidate_ordering.is_pow2(5) == False
    assert candidate_ordering.is_pow2(32) == True
    assert candidate_ordering.is_pow2(6) == False

    assert candidate_ordering.is_mult_simd_num(6, 4) == False
    assert candidate_ordering.is_mult_simd_num(8, 4) == True

    ai = candidate_ordering.arith_intensity(2, 3, 4)
    expected = (2 * 2 * 3 * 4) / (2 * (2 * 3 + 3 * 4 + 2 * 4))
    assert math.isclose(ai, expected, rel_tol=1e-9)


def test_reorder_assignments(
    target_info: iree_gpu.TargetInfo,
    sample_knobs: list[Optional[common.KnobAssignment]],
) -> None:
    expected_order = [0, 1, 2]
    assert (
        candidate_ordering.reorder_assignments(
            target_info=target_info,
            knobs=sample_knobs,
            strategy=candidate_ordering.CandidateOrderKind.no_sort,
        )
        == expected_order
    )

    expected_order = [2, 0, 1]
    assert (
        candidate_ordering.reorder_assignments(
            target_info=target_info,
            knobs=sample_knobs,
            strategy=candidate_ordering.CandidateOrderKind.heuristic,
        )
        == expected_order
    )

    expected_order = [0, 2, 1]
    assert (
        candidate_ordering.reorder_assignments(
            knobs=sample_knobs,
            strategy=candidate_ordering.CandidateOrderKind.heuristic,
            key_fn=lambda knob: knob.tile_n,
        )
        == expected_order
    )

    knobs: list[Optional[common.KnobAssignment]] = [None, None, None]
    assert (
        candidate_ordering.reorder_assignments(
            target_info=target_info,
            knobs=knobs,
            strategy=candidate_ordering.CandidateOrderKind.shuffle,
        )
        != []
    )

    knobs = []
    assert (
        candidate_ordering.reorder_assignments(
            target_info=target_info,
            knobs=knobs,
            strategy=candidate_ordering.CandidateOrderKind.shuffle,
        )
        == []
    )


def test_init_tuning_records(
    sample_knobs: list[Optional[common.KnobAssignment]],
) -> None:
    tr0 = candidate_ordering.TuningRecord(
        gen_id=0,
        candidate_id=0,
        to_compile=True,
        to_benchmark=True,
    )
    tr1 = candidate_ordering.TuningRecord(
        gen_id=2,
        candidate_id=1,
        knob=sample_knobs[2],
    )
    tr2 = candidate_ordering.TuningRecord(
        gen_id=0,
        candidate_id=2,
        knob=sample_knobs[0],
    )
    tr3 = candidate_ordering.TuningRecord(
        gen_id=1,
        candidate_id=3,
        knob=sample_knobs[1],
    )
    sorted_order = [2, 0, 1]
    tuning_records = candidate_ordering.init_tuning_records(sample_knobs, sorted_order)

    expected = [tr0, tr1, tr2, tr3]

    assert tuning_records == expected


def test_flatten_records(
    sample_knobs: list[Optional[common.KnobAssignment]],
):
    tr0 = candidate_ordering.TuningRecord(
        gen_id=0,
        candidate_id=0,
        to_compile=True,
        to_benchmark=True,
    )
    tr1 = candidate_ordering.TuningRecord(
        gen_id=2,
        candidate_id=1,
        knob=sample_knobs[2],
        to_compile=True,
        benchmark_device_id="hip://2",
        benchmark_queue_position=1,
        baseline_benchmark_time_us=123.4,
        benchmark_speedup=1.5,
    )
    tr2 = candidate_ordering.TuningRecord(
        gen_id=1,
        candidate_id=2,
        knob=sample_knobs[1],
        to_benchmark=True,
        benchmark_time_us=153.56,
    )
    sample_tuning_records = [tr0, tr1, tr2]

    headers, rows = candidate_ordering.flatten_records(sample_tuning_records)

    expected_headers = [
        "gen_id",
        "candidate_id",
        "to_compile",
        "compile_status",
        "to_benchmark",
        "benchmark_device_id",
        "benchmark_queue_position",
        "benchmark_status",
        "baseline_benchmark_time_us",
        "benchmark_time_us",
        "benchmark_speedup",
        "benchmark_rank_order",
        "knob.M",
        "knob.N",
        "knob.K",
        "knob.tile_m",
        "knob.tile_n",
        "knob.tile_k",
        "knob.wg_x",
        "knob.wg_y",
        "knob.wg_z",
        "knob.subgroup_m_cnt",
        "knob.subgroup_n_cnt",
        "knob.intrinsic_mn",
        "knob.intrinsic_k",
        "knob.subgroup_m",
        "knob.subgroup_n",
        "knob.subgroup_k",
    ]
    assert headers == expected_headers

    expected_rows = [
        {
            "baseline_benchmark_time_us": None,
            "benchmark_device_id": None,
            "benchmark_queue_position": None,
            "benchmark_rank_order": None,
            "benchmark_speedup": None,
            "benchmark_status": False,
            "benchmark_time_us": None,
            "candidate_id": 0,
            "compile_status": False,
            "gen_id": 0,
            "knob": None,
            "to_benchmark": True,
            "to_compile": True,
        },
        {
            "baseline_benchmark_time_us": 123.4,
            "benchmark_device_id": "hip://2",
            "benchmark_queue_position": 1,
            "benchmark_rank_order": None,
            "benchmark_speedup": 1.5,
            "benchmark_status": False,
            "benchmark_time_us": None,
            "candidate_id": 1,
            "compile_status": False,
            "gen_id": 2,
            "knob.K": 1280,
            "knob.M": 2048,
            "knob.N": 10240,
            "knob.intrinsic_k": 16,
            "knob.intrinsic_mn": 16,
            "knob.subgroup_k": 0,
            "knob.subgroup_m": 0,
            "knob.subgroup_m_cnt": 2,
            "knob.subgroup_n": 0,
            "knob.subgroup_n_cnt": 4,
            "knob.tile_k": 16,
            "knob.tile_m": 64,
            "knob.tile_n": 256,
            "knob.wg_x": 256,
            "knob.wg_y": 2,
            "knob.wg_z": 1,
            "to_benchmark": False,
            "to_compile": True,
        },
        {
            "baseline_benchmark_time_us": None,
            "benchmark_device_id": None,
            "benchmark_queue_position": None,
            "benchmark_rank_order": None,
            "benchmark_speedup": None,
            "benchmark_status": False,
            "benchmark_time_us": 153.56,
            "candidate_id": 2,
            "compile_status": False,
            "gen_id": 1,
            "knob.K": 1280,
            "knob.M": 2048,
            "knob.N": 10240,
            "knob.intrinsic_k": 16,
            "knob.intrinsic_mn": 16,
            "knob.subgroup_k": 0,
            "knob.subgroup_m": 0,
            "knob.subgroup_m_cnt": 1,
            "knob.subgroup_n": 0,
            "knob.subgroup_n_cnt": 5,
            "knob.tile_k": 80,
            "knob.tile_m": 64,
            "knob.tile_n": 320,
            "knob.wg_x": 320,
            "knob.wg_y": 1,
            "knob.wg_z": 1,
            "to_benchmark": True,
            "to_compile": False,
        },
    ]
    assert rows == expected_rows
