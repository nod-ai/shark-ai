# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
import iree.runtime
import subprocess
import json
import sys
import pandas
from pathlib import Path
import os
from os import PathLike


def _run_program(
    args: tuple[str],
):
    process_result = subprocess.run(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    out = process_result.stdout.decode()
    err = process_result.stderr.decode()

    if process_result.returncode != 0:
        raise RuntimeError(f"stderr:\n{err}\nstdout:\n{out}")

    if err != "":
        print(err, file=sys.stderr)

    return out


def iree_benchmark_module(
    cli_args: tuple[str],
):
    args = [iree.runtime.benchmark_exe()] + cli_args
    return _run_program(args=args)


def google_benchmark_compare_path() -> str:
    return os.path.abspath(
        Path(__file__).parent.parent.parent.parent
        / "third_party"
        / "benchmark"
        / "tools"
        / "compare.py"
    )


def iree_benchmark_compare(cli_args: tuple[str]):
    args = [google_benchmark_compare_path()] + cli_args
    return _run_program(args=args)


def _get_benchmark_comparison_aggregate_real_time(
    benchmark_comparison_result_json: dict[str, Any], aggregate: str
) -> tuple[float, float, str]:
    real_time = [
        (
            benchmark["measurements"][0]["real_time"],
            benchmark["measurements"][0]["real_time_other"],
            benchmark["time_unit"],
        )
        for benchmark in benchmark_comparison_result_json
        if "aggregate_name" in benchmark and benchmark["aggregate_name"] == aggregate
    ]
    assert len(real_time) == 1
    return real_time[0]


def _assert_contender_aggregate_real_time_is_not_worse(
    benchmark_comparison_result_json: dict[str, Any], aggregate: str
):
    real_time = _get_benchmark_comparison_aggregate_real_time(
        benchmark_comparison_result_json, aggregate
    )
    baseline_real_time, contender_real_time, time_unit = real_time
    if baseline_real_time < contender_real_time:
        raise AssertionError(
            f"Benchmark contender {aggregate} "
            f"real time {contender_real_time} {time_unit} "
            f"is worse than baseline {baseline_real_time} {time_unit}."
        )


def iree_benchmark_assert_contender_is_not_worse(
    benchmark_comparison_result_json: dict[str, Any], alpha: float = 0.05
):
    """If contender is not from the same distribution as baseline, assert that and
    that its median and mean is not worse.

    Arguments
    ---------
    alpha: acceptance/significance threshold probability that the two benchmark sample
    sets are from the same distribution. Meaning they are not different."""
    time_pvalue = [
        b["utest"]["time_pvalue"]
        for b in benchmark_comparison_result_json
        if "utest" in b and "time_pvalue" in b["utest"]
    ]
    assert len(time_pvalue) == 1
    time_pvalue = time_pvalue[0]
    if alpha <= time_pvalue:
        # The benchmarks are from the same distribution.
        return

    _assert_contender_aggregate_real_time_is_not_worse(
        benchmark_comparison_result_json, "mean"
    )
    _assert_contender_aggregate_real_time_is_not_worse(
        benchmark_comparison_result_json, "median"
    )
