#!/usr/bin/env python3
# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import csv
import glob
import os
import shlex
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple


class TimingStats(NamedTuple):
    min: float | str = "N.A."
    max: float | str = "N.A."
    mean: float | str = "N.A."
    stddev: float | str = "N.A."
    iter: int | str = "N.A."
    dispatch_count: int | str = "N.A."


class CommandResult(NamedTuple):
    stats: TimingStats
    timed_out: bool = False
    failed: bool = False
    skipped: bool = False
    succeeded: bool = False


ALL_METRICS = ["min", "max", "mean", "stddev", "iter", "dispatch_count"]


def parse_rocprof_csv(output_dir: Path, iter_count: int) -> TimingStats:
    kernel_trace_files = list(output_dir.rglob("*kernel_trace.csv"))

    if not kernel_trace_files:
        return TimingStats()

    durations = []

    for csv_file in kernel_trace_files:
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                # Each row represents a single kernel dispatch
                for row in reader:
                    # Sometimes, we may see non-async_dispatch rows in the CSV file.
                    # It is not consistently reproducible, so we have this check
                    # to filter out non-async_dispatch rows.
                    has_async_dispatch = any(
                        "async_dispatch" in str(value).lower() for value in row.values()
                    )
                    if (
                        has_async_dispatch
                        and "Start_Timestamp" in row
                        and "End_Timestamp" in row
                    ):
                        start = float(row["Start_Timestamp"])
                        end = float(row["End_Timestamp"])
                        # Convert from nanoseconds to microseconds
                        duration_us = (end - start) / 1000.0
                        durations.append(duration_us)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse rocprof CSV file {csv_file}: {e}"
            ) from e

    if not durations:
        return TimingStats()

    # Group dispatches by iteration
    # Total dispatches / iter_count = dispatches per iteration
    total_dispatches = len(durations)

    if iter_count > 0 and total_dispatches >= iter_count:
        assert (
            total_dispatches % iter_count == 0
        ), "Total dispatches must be divisible by iter_count"
        dispatches_per_iter = total_dispatches // iter_count
        # Group consecutive dispatches into iterations and sum their durations
        iteration_durations = []
        for i in range(iter_count):
            start_idx = i * dispatches_per_iter
            end_idx = start_idx + dispatches_per_iter
            iter_sum = sum(durations[start_idx:end_idx])
            iteration_durations.append(iter_sum)

        # Compute statistics across iterations
        min_time = min(iteration_durations)
        max_time = max(iteration_durations)
        mean_time = statistics.mean(iteration_durations)
        stddev = (
            statistics.stdev(iteration_durations)
            if len(iteration_durations) > 1
            else 0.0
        )
        iter_count_result = len(iteration_durations)
        dispatch_count = dispatches_per_iter
    else:
        raise RuntimeError(
            f">>> ERROR: Invalid iter_count: {iter_count} or total_dispatches: {total_dispatches} < iter_count."
        )

    return TimingStats(
        min=min_time,
        max=max_time,
        mean=mean_time,
        stddev=stddev,
        iter=iter_count_result,
        dispatch_count=dispatch_count,
    )


def run_profiled_command(
    command: str,
    driver_path: str,
    output_dir: Path | None,
    rocprof_args: list[str],
    verbose: bool,
    cmd_num: int,
    timeout: int,
) -> CommandResult:

    driver_args = command.split()
    if not driver_args:
        if verbose:
            print(f">>> Failed to parse command: {command}")
        return CommandResult(TimingStats(), failed=True)

    iter_count = 1
    if "--iter" in driver_args:
        iter_idx = driver_args.index("--iter")
        if iter_idx + 1 < len(driver_args):
            iter_count = int(driver_args[iter_idx + 1])

    driver_cmd = [driver_path] + driver_args

    # Use either temporary directory or persistent directory
    if output_dir is None:
        tmpdir_context = tempfile.TemporaryDirectory()
        cmd_output_dir = Path(tmpdir_context.__enter__())
    else:
        tmpdir_context = None
        cmd_output_dir = output_dir / f"command_{cmd_num}"
        cmd_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rocprof_cmd = (
            [
                "rocprofv3",
                "--output-format",
                "csv",
                "--output-directory",
                str(cmd_output_dir),
            ]
            + rocprof_args
            + ["--"]
            + driver_cmd
        )

        if verbose:
            print(f">>> {shlex.join(rocprof_cmd)}\n")

        timeout_val = None if timeout == -1 else timeout
        result = subprocess.run(
            rocprof_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_val,
        )

        if verbose and result.stdout:
            print(result.stdout)

        stats = parse_rocprof_csv(cmd_output_dir, iter_count)
        print(
            f">>> Stats: min={stats.min:.2f}(us), max={stats.max:.2f}(us), mean={stats.mean:.2f}(us), iter={stats.iter}, dispatch_count={stats.dispatch_count}"
        )

        return CommandResult(stats, succeeded=True)

    except subprocess.TimeoutExpired:
        if verbose:
            print(f">>> Command timed out after {timeout} seconds")
        return CommandResult(TimingStats(), timed_out=True)
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f">>> Command failed with exit code {e.returncode}")
            if e.stderr:
                print(f">>> stderr: {e.stderr}")
        return CommandResult(TimingStats(), failed=True)
    except Exception as e:
        if verbose:
            print(f">>> Exception: {e}")
        return CommandResult(TimingStats(), failed=True)
    finally:
        # Cleanup temporary directory if used
        if tmpdir_context is not None:
            tmpdir_context.__exit__(None, None, None)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="""
Run Fusilli benchmarks with rocprofv3 profiling and aggregate results.

Commands are read from a file (one per line). Each command is run through
rocprofv3, and timing statistics are collected and written to a CSV file.

Command format example:
  conv --bf16 -n 16 -c 64 -H 48 -W 32 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC --spatial_dim 2

Commands starting with [SKIP] will be skipped:
  [SKIP] conv --bf16 -n 128 -c 384 -H 24 -W 48 -k 384 -y 1 -x 3 ...

The script will:
  1. Run each command under rocprofv3 (or skip if prefixed with [SKIP])
  2. Apply timeout to each command (default: 60 seconds)
  3. Extract timing statistics from rocprof CSV outputs
  4. Aggregate results into a single output CSV
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--commands-file",
        "-f",
        type=str,
        required=True,
        help="File containing benchmark commands (one per line)",
    )

    parser.add_argument(
        "--csv",
        "-o",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for aggregated results (default: benchmark_results.csv)",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default=None,
        help="Directory to store rocprof outputs. If not provided, uses temporary directories with auto-cleanup.",
    )

    script_dir = Path(__file__).parent.parent.absolute()
    default_driver = (
        script_dir / "build" / "bin" / "benchmarks" / "fusilli_benchmark_driver"
    )
    parser.add_argument(
        "--driver",
        "-D",
        type=str,
        default=str(default_driver),
        help=f"Path to fusilli_benchmark_driver binary (default: {default_driver})",
    )

    parser.add_argument(
        "--rocprof-args",
        "-r",
        type=str,
        default="--runtime-trace",
        help="Arguments for rocprofv3 (default: --runtime-trace)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=30,
        help="Timeout in seconds for each command (default: 30, use -1 for no timeout)",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    commands_file = Path(args.commands_file)
    if not commands_file.exists():
        print(f"Error: Commands file not found: {commands_file}")
        return 1

    with open(commands_file, "r") as f:
        commands = [
            stripped
            for line in f.readlines()
            if (stripped := line.strip()) and not stripped.startswith("#")
        ]

    if not commands:
        print("Error: No commands found in file")
        return 1

    print(f"Found {len(commands)} commands")

    # Use tempdir by default unless output_dir is explicitly provided
    output_dir = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rocprof_args = (
        args.rocprof_args.split() if args.rocprof_args else ["--runtime-trace"]
    )

    if args.verbose:
        print(f"Rocprof args: {' '.join(rocprof_args)}")
        if output_dir is None:
            print("Using temporary directories (auto-cleanup)")
        else:
            print(f"Output directory: {output_dir.absolute()}")
        timeout_str = "no timeout" if args.timeout == -1 else f"{args.timeout} seconds"
        print(f"Timeout: {timeout_str}")
        print(f"Results will be written to: {args.csv}\n")

    csv_file = csv.writer(open(args.csv, "w", newline=""))
    csv_headers = ["command"]
    for metric in ALL_METRICS:
        if metric in ["min", "max", "mean", "stddev"]:
            csv_headers.append(f"{metric} (us)")
        else:
            csv_headers.append(metric)
    csv_file.writerow(csv_headers)

    cmd_count = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0
    timeout_count = 0

    for command in commands:
        cmd_count += 1

        # Check if command should be skipped
        skip_prefix = "[SKIP]"
        is_skipped = command.startswith(skip_prefix)

        if is_skipped:
            display_command = command[len(skip_prefix) :].strip()
            print(f"\n{'='*80}")
            print(f"Skipping command {cmd_count}/{len(commands)}:\n{display_command}")
            print(f"{'='*80}")
            # Create a result with default (N.A.) stats for skipped commands
            result = CommandResult(TimingStats(), skipped=True)

        else:
            print(f"\n{'='*80}")
            print(f"Running command {cmd_count}/{len(commands)}:\n{command}")
            print(f"{'='*80}")
            # Run the command and collect statistics
            result = run_profiled_command(
                command,
                args.driver,
                output_dir,
                rocprof_args,
                args.verbose,
                cmd_count,
                args.timeout,
            )

        stats = result.stats
        csv_row = [command]
        for metric in ALL_METRICS:
            value = getattr(stats, metric)
            csv_row.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        csv_file.writerow(csv_row)

        if result.succeeded:
            assert isinstance(stats.mean, float)
            success_count += 1
        elif result.skipped:
            print(">>> Skipped")
            skipped_count += 1
        elif result.timed_out:
            print(f">>> Timed out")
            timeout_count += 1
        elif result.failed:
            print(">>> Failed")
            failed_count += 1
        else:
            raise RuntimeError(f"Unknown result: {result}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total commands: {cmd_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Timed out: {timeout_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Results CSV: {args.csv}")
    if output_dir is not None:
        print(f"Rocprof outputs: {output_dir.absolute()}")
    print(f"{'='*80}\n")

    return 0 if (failed_count + timeout_count == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
