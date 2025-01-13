# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Quite a few of our tests are best tested in a dedicated process. We write
# those as examples and launch them here.

from pathlib import Path
import subprocess
import sys
import os

project_dir = Path(__file__).parent.parent.parent
example_dir = project_dir / "examples" / "python"


def run_example(path: Path):
    subprocess.check_call([sys.executable, str(path)], timeout=60)


def test_async_basic_asyncio():
    run_example(example_dir / "async" / "basic_asyncio.py")


def test_async_device_sync():
    run_example(example_dir / "async" / "device_sync.py")


def test_async_queue():
    run_example(example_dir / "async" / "queue.py")


def test_async_process():
    run_example(example_dir / "async" / "process.py")
