# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from typing import AnyStr, IO
import argparse
import datetime
import shutil
import subprocess
import sys
import time


def collect_process_status(pid: int, out_file: IO[AnyStr]):
    out_file.write(f"Time: {datetime.datetime.now()}\n")

    try:
        with open(f"/proc/{pid}/status", "r") as f:
            shutil.copyfileobj(f, out_file)
    except FileNotFoundError:
        # Even before doing subprocess.Popen.wait from the parent to collect the
        # zombie child the file get deleted.
        # Linux's wait3/wait4 are not good enough as they don't report the peak virtual
        # memory. Thus GNU time is also unable to report it.
        # See
        # https://linux.die.net/man/2/wait3
        # https://linux.die.net/man/2/getrusage
        pass


def collect_status(
    process: subprocess.Popen,
    out_file: IO[AnyStr] | str | Path | None = None,
    period: float | None = None,
):
    if out_file is None or (isinstance(out_file, (str, Path)) and "-" == f"{out_file}"):
        out_file = sys.stderr
    elif isinstance(out_file, (str, Path)):
        out_file = open(out_file, "w")
    last_collection_time = time.monotonic()
    while process.returncode is None:
        next_collection_time = last_collection_time + period
        timeout = max(0, next_collection_time - time.monotonic())
        try:
            process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        if next_collection_time <= time.monotonic():
            collect_process_status(process.pid, out_file)
            last_collection_time = time.monotonic()

    collect_process_status(process.pid, out_file)
    process.wait()


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run and collect stats of a process on linux using /proc/<pid>/status. "
            "At the end of execution always collects before harvesting the process. "
            "Unfortunately, even before waiting on the process the /proc/<pid>/status file will be remove. "
            "This will cause the final read to be missed. "
            "Due to that this tool is really just a periodic monitoring. "
            "This allow you to measure measure the peak virtual memory (VmPeak) and "
            "peak physical memory (VmHWM). GNU time can do that but will report only VmHWM."
        )
    )
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="The time interval between collections in seconds.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="File path where to collect the process status. Defaults to stderr.",
    )
    parser.add_argument("cmd", nargs="+", help="Command with arguments to execute.")
    parsed_args = parser.parse_args(args)

    process = subprocess.Popen(parsed_args.cmd)
    collect_status(process, out_file=parsed_args.out_file, period=parsed_args.period)


if __name__ == "__main__":
    main()
