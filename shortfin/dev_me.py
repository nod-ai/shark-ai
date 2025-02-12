#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# dev_me.py
#
# This is an opinionated development environment setup procedure aimed at
# making core contributors on the same golden path. It is not the only way
# to develop this project.
#
# First time build usage:
#   rm -Rf build  # Start with a fresh build dir
#   python dev_me.py [--cmake=/path/to/cmake] [--clang=/path/to/clang] \
#     [--iree=/path/to/iree] [--asan] [--build-type=Debug] \
#     [--no-tracing]
#
# Subsequent build:
#   ./dev_me.py
#
# This will perform an editable install into the used python with both
# default and tracing packages installed. After the initial build, ninja
# can be invoked directly under build/cmake/default or build/cmake/tracy.
# This can be done automatically just by running dev_me.py in a tree with
# an existing build directory.
#
# By default, if there is an iree source dir adjacent to this parent repository,
# that will be used (so you can just directly edit IREE runtime code and build.
# Otherwise, the shortfin build will download a pinned IREE source tree.

import argparse
import importlib
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
import sysconfig

try:
    from packaging.version import Version
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"'packaging' package not installed and required: Install with:\n"
        f"  {sys.executable} -m pip install packaging"
    )


CMAKE_REQUIRED_VERSION = Version("3.29")
PYTHON_REQUIRED_VERSION = Version("3.12")
CLANG_REQUIRED_VERSION = Version("16")
SETUPTOOLS_REQUIRED_VERSION = Version("61.0")


class EnvInfo:
    def __init__(self, args):
        self.this_dir = Path(__file__).parent
        self.python_exe = sys.executable
        self.python_version = Version(".".join(str(v) for v in sys.version_info[1:2]))
        self.debug = bool(sysconfig.get_config_var("Py_DEBUG"))
        self.asan = "-fsanitize=address" in sysconfig.get_config_var("PY_LDFLAGS")
        self.gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        self.cmake_exe, self.cmake_version = self.find_cmake(args)
        self.ninja_exe = shutil.which("ninja")
        self.clang_exe, self.clang_version = self.find_clang(args)
        self.iree_dir = self.find_iree(args)
        self.setuptools_version = self.find_package_version("setuptools")
        self.wheel_version = self.find_package_version("wheel")

        self.configured_dirs = []
        self.add_configured(self.this_dir / "build" / "cmake" / "default")
        self.add_configured(self.this_dir / "build" / "cmake" / "tracy")

    def add_configured(self, path: Path):
        probe = path / "CMakeCache.txt"
        if probe.exists():
            self.configured_dirs.append(path)

    def find_cmake(self, args):
        paths = []
        if args.cmake:
            paths.append(str(args.cmake))
        else:
            default_cmake = shutil.which("cmake")
            if default_cmake:
                paths.append(default_cmake)
        for cmake_path in paths:
            try:
                cmake_output = subprocess.check_output(
                    [cmake_path, "--version"]
                ).decode()
            except:
                continue
            if m := re.search("cmake version (.+)", cmake_output):
                return cmake_path, Version(m.group(1))
        return None, None

    def find_clang(self, args):
        if args.clang:
            clang_exe = args.clang
        else:
            clang_exe = shutil.which("clang")
            if not clang_exe:
                return None, None
        try:
            clang_output = subprocess.check_output([clang_exe, "--version"]).decode()
        except:
            return None, None
        if m := re.search(r"clang version ([0-9\.]+)", clang_output):
            return clang_exe, Version(m.group(1))
        return None, None

    def find_iree(self, args):
        iree_dir = args.iree
        if not iree_dir:
            # See if a sibling iree directory exists.
            iree_dir = self.this_dir.parent.parent / "iree"
            if (iree_dir / "CMakeLists.txt").exists():
                return str(iree_dir)
        if not iree_dir.exists():
            print(f"ERROR: --iree={iree_dir} directory does not exist")
            sys.exit(1)
        return str(iree_dir)

    def find_package_version(self, package_name: str) -> Version | None:
        try:
            m = importlib.import_module(package_name)
        except ModuleNotFoundError:
            return None
        return Version(m.__version__)

    def check_prereqs(self, args):
        if self.cmake_version is None or self.cmake_version < CMAKE_REQUIRED_VERSION:
            print(
                f"ERROR: cmake not found or of an insufficient version: {self.cmake_exe}"
            )
            print(f"  Required: {CMAKE_REQUIRED_VERSION}, Found: {self.cmake_version}")
            print(f"  Configure explicitly with --cmake=")
            sys.exit(1)
        if self.python_version < PYTHON_REQUIRED_VERSION:
            print(f"ERROR: python version too old: {self.python_exe}")
            print(
                f"  Required: {PYTHON_REQUIRED_VERSION}, Found: {self.python_version}"
            )
            sys.exit(1)
        if self.clang_exe and self.clang_version < CLANG_REQUIRED_VERSION:
            print(f"WARNING: clang version too old: {self.clang_exe}")
            print(f"  REQUIRED: {CLANG_REQUIRED_VERSION}, Found {self.clang_version}")
        elif not self.clang_exe:
            print(f"WARNING: Building the project with clang is highly recommended")
            print(f"  (pass --clang= to select clang)")

        if args.asan and not self.asan:
            print(
                f"ERROR: An ASAN build was requested but python was not built with ASAN support"
            )
            sys.exit(1)

        if (
            self.setuptools_version is None
            or self.setuptools_version < SETUPTOOLS_REQUIRED_VERSION
        ):
            print(
                f"ERROR: 'setuptools' packaging is not installed or too old. "
                f"Found {self.setuptools_version}, Need {SETUPTOOLS_REQUIRED_VERSION}"
            )
            sys.exit(1)
        if self.wheel_version is None:
            print(f"'wheel' package is not installed")
            sys.exit(1)

    def __repr__(self):
        report = [
            f"python: {self.python_exe}",
            f"debug: {self.debug}",
            f"asan: {self.asan}",
            f"gil_disabled: {self.gil_disabled}",
            f"cmake: {self.cmake_exe} ({self.cmake_version})",
            f"ninja: {self.ninja_exe}",
            f"clang: {self.clang_exe} ({self.clang_version})",
            f"iree: {self.iree_dir}",
            f"setuptools: {self.setuptools_version}",
            f"wheel: {self.wheel_version}",
        ]
        return "\n".join(report)


def main(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog="shortfin dev", description="Shortfin dev setup helper"
    )
    parser.add_argument("--cmake", type=Path, help="CMake path")
    parser.add_argument("--clang", type=Path, help="Clang path")
    parser.add_argument("--iree", type=Path, help="Path to IREE source checkout")
    parser.add_argument("--asan", action="store_true", help="Build with ASAN support")
    parser.add_argument(
        "--no-tracing", action="store_true", help="Disable IREE tracing build"
    )
    parser.add_argument(
        "--build-type", default="Debug", help="CMake build type (default Debug)"
    )
    args = parser.parse_args(argv)
    env_info = EnvInfo(args)

    if env_info.configured_dirs:
        print("First time configure...")
        build_mode(env_info)
    else:
        configure_mode(env_info, args)


def configure_mode(env_info: EnvInfo, args):
    print("Environment info:")
    print(env_info)
    env_info.check_prereqs(args)

    env_vars = {
        "SHORTFIN_DEV_MODE": "ON",
        "SHORTFIN_CMAKE_BUILD_TYPE": args.build_type,
        "SHORTFIN_ENABLE_ASAN": "ON" if args.asan else "OFF",
        "SHORTFIN_CMAKE": env_info.cmake_exe,
    }
    if env_info.iree_dir:
        env_vars["SHORTFIN_IREE_SOURCE_DIR"] = env_info.iree_dir
    if env_info.clang_exe:
        env_vars["CC"] = env_info.clang_exe
        env_vars["CXX"] = f"{env_info.clang_exe}++"
        env_vars["CMAKE_LINKER_TYPE"] = "LLD"
    env_vars["SHORTFIN_ENABLE_TRACING"] = "OFF" if args.no_tracing else "ON"

    print("Executing setup:")
    setup_args = [
        env_info.python_exe,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-v",
        "-e",
        str(env_info.this_dir),
    ]
    print(f"{' '.join('='.join(str(kv)) for kv in env_vars.items())} \\")
    print(f"  {' '.join(setup_args)}")
    actual_env_vars = dict(os.environ)
    actual_env_vars.update(env_vars)
    subprocess.check_call(setup_args, cwd=env_info.this_dir, env=actual_env_vars)
    print("You are now DEV'd!")


def build_mode(env_info: EnvInfo):
    print("Building")
    for build_dir in env_info.configured_dirs:
        subprocess.check_call(
            [env_info.cmake_exe, "--build", str(build_dir), "--verbose"]
        )


if __name__ == "__main__":
    main(sys.argv[1:])
