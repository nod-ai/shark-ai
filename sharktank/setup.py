# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
from pathlib import Path
import subprocess
from setuptools.command.build_py import build_py as _build_py

from setuptools import setup

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))

# Setup and get version information.
VERSION_FILE = os.path.join(SETUPPY_DIR, "version.json")
VERSION_FILE_LOCAL = os.path.join(SETUPPY_DIR, "version_local.json")


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


try:
    version_info = load_version_info(VERSION_FILE_LOCAL)
except FileNotFoundError:
    print("version_local.json not found. Default to dev build")
    version_info = load_version_info(VERSION_FILE)

PACKAGE_VERSION = version_info.get("package-version")
print(f"Using PACKAGE_VERSION: '{PACKAGE_VERSION}'")


class BuildCustomKernels(_build_py):
    def run(self):
        hip_kernels_dir = os.path.join(os.path.dirname(__file__), "hip_kernels")
        build_dir = os.path.join(hip_kernels_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(["cmake", ".."], cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "all_hsaco_kernels"], cwd=build_dir
        )
        super().run()


setup(
    version=f"{PACKAGE_VERSION}",
    cmdclass={
        "build_py": BuildCustomKernels,
    },
)
