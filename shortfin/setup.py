# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import shutil
import subprocess
import sys
import traceback
from distutils.command.build import build as _build
from distutils.core import setup, Extension
from pathlib import Path
from setuptools import find_namespace_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext


def get_env_boolean(name: str, default_value: bool = False) -> bool:
    svalue = os.getenv(name)
    if svalue is None:
        return default_value
    svalue = svalue.upper()
    if svalue in ["1", "ON", "TRUE"]:
        return True
    elif svalue in ["0", "OFF", "FALSE"]:
        return False
    else:
        print(f"WARNING: {name} env var cannot be interpreted as a boolean value")
        return default_value


def get_env_cmake_option(name: str, default_value: bool = False) -> str:
    svalue = os.getenv(name)
    if not svalue:
        svalue = "ON" if default_value else "OFF"
    return f"-D{name}={svalue}"


def add_env_cmake_setting(
    args, env_name: str, cmake_name=None, default_value=None
) -> str:
    svalue = os.getenv(env_name)
    if svalue is None and default_value is not None:
        svalue = default_value
    if svalue is not None:
        if not cmake_name:
            cmake_name = env_name
        args.append(f"-D{cmake_name}={svalue}")


def combine_dicts(*ds):
    result = {}
    for d in ds:
        result.update(d)
    return result


# This file can be generated into the build directory to allow an arbitrary
# CMake built version of the project to be installed into a venv for development.
# This can be detected if the CPP_PREBUILT global contains the string
# "TRUE", which will be the case if generated.
CPP_PREBUILT = "@SHORTFIN_PYTHON_CPP_PREBUILT@"
CPP_PREBUILT_SOURCE_DIR = "@libshortfin_SOURCE_DIR@"
CPP_PREBUILT_BINARY_DIR = "@libshortfin_BINARY_DIR@"

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))
CMAKE_EXE = os.getenv("SHORTFIN_CMAKE", "cmake")


def is_cpp_prebuilt():
    return CPP_PREBUILT == "TRUE"


DEV_MODE = False
ENABLE_TRACY = get_env_boolean("SHORTFIN_ENABLE_TRACING", False)

if ENABLE_TRACY:
    print(
        "*** Enabling Tracy instrumentation (disable with SHORTFIN_ENABLE_TRACING=OFF)",
    )
else:
    print(
        "*** Tracy instrumentation not enabled (enable with SHORTFIN_ENABLE_TRACING=ON)",
    )


if is_cpp_prebuilt():
    print("setup.py running in pre-built mode:")
    SOURCE_DIR = Path(CPP_PREBUILT_SOURCE_DIR)
    BINARY_DIR = Path(CPP_PREBUILT_BINARY_DIR)
    CMAKE_DEFAULT_BUILD_DIR = BINARY_DIR
    CMAKE_TRACY_BUILD_DIR = BINARY_DIR
else:
    print("setup.py running in cmake build mode:")
    # setup.py is in the source directory.
    SOURCE_DIR = Path(SETUPPY_DIR)
    BINARY_DIR = Path(os.path.join(SETUPPY_DIR, "build"))
    # TODO: Should build default and tracing version to different dirs.
    CMAKE_DEFAULT_BUILD_DIR = BINARY_DIR / "cmake" / "default"
    CMAKE_TRACY_BUILD_DIR = BINARY_DIR / "cmake" / "tracy"
    DEV_MODE = get_env_boolean("SHORTFIN_DEV_MODE")

print(f"  SOURCE_DIR = {SOURCE_DIR}")
print(f"  BINARY_DIR = {BINARY_DIR}")

if DEV_MODE:
    print(f"  DEV MODE ENABLED: Building debug with clang/lld and other dev settings")

# Due to a quirk of setuptools, that package_dir map must only contain
# paths relative to the directory containing setup.py. Why? No one knows.
REL_SOURCE_DIR = Path(os.path.relpath(SOURCE_DIR, SETUPPY_DIR))
REL_BINARY_DIR = Path(os.path.relpath(BINARY_DIR, SETUPPY_DIR))
REL_CMAKE_DEFAULT_BUILD_DIR = Path(
    os.path.relpath(CMAKE_DEFAULT_BUILD_DIR, SETUPPY_DIR)
)
REL_CMAKE_TRACY_BUILD_DIR = Path(os.path.relpath(CMAKE_TRACY_BUILD_DIR, SETUPPY_DIR))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


class NoopBuildExtension(_build_ext):
    def build_extension(self, ext):
        ...

    def copy_extensions_to_source(self, *args, **kwargs):
        ...


# Setup and get version information.
VERSION_FILE = os.path.join(REL_SOURCE_DIR, "version.json")
VERSION_FILE_LOCAL = os.path.join(REL_SOURCE_DIR, "version_local.json")


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


def maybe_nuke_cmake_cache(cmake_build_dir):
    # From run to run under pip, we can end up with different paths to ninja,
    # which isn't great and will confuse cmake. Detect if the location of
    # ninja changes and force a cache flush.
    ninja_path = ""
    try:
        import ninja
    except ModuleNotFoundError:
        pass
    else:
        ninja_path = ninja.__file__
    expected_stamp_contents = f"{sys.executable}\n{ninja_path}"

    # In order to speed things up on CI and not rebuild everything, we nuke
    # the CMakeCache.txt file if the path to the Python interpreter changed.
    # Ideally, CMake would let us reconfigure this dynamically... but it does
    # not (and gets very confused).
    PYTHON_STAMP_FILE = os.path.join(cmake_build_dir, "python_stamp.txt")
    if os.path.exists(PYTHON_STAMP_FILE):
        with open(PYTHON_STAMP_FILE, "rt") as f:
            actual_stamp_contents = f.read()
            if actual_stamp_contents == expected_stamp_contents:
                # All good.
                return

    # Mismatch or not found. Clean it.
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
        print("Removing CMakeCache.txt because Python version changed")
        os.remove(cmake_cache_file)

    # And write.
    with open(PYTHON_STAMP_FILE, "wt") as f:
        f.write(expected_stamp_contents)


def build_cmake_configuration(CMAKE_BUILD_DIR: Path, extra_cmake_args=[]):
    # Build extension using cmake.
    cfg = os.getenv("SHORTFIN_CMAKE_BUILD_TYPE", "Debug" if DEV_MODE else "Release")

    # Configure CMake.
    os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)
    if not DEV_MODE:
        maybe_nuke_cmake_cache(CMAKE_BUILD_DIR)
    print(f"CMake build dir: {CMAKE_BUILD_DIR}")
    cmake_args = [
        "-GNinja",
        "-Wno-dev",
        "--log-level=VERBOSE",
        "-DSHORTFIN_BUNDLE_DEPS=ON",
        f"-DCMAKE_BUILD_TYPE={cfg}",
        "-DSHORTFIN_BUILD_PYTHON_BINDINGS=ON",
        "-DSHORTFIN_BUILD_TESTS=OFF",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ] + extra_cmake_args

    if DEV_MODE:
        if not os.getenv("CC"):
            cmake_args.append("-DCMAKE_C_COMPILER=clang")
        if not os.getenv("CXX"):
            cmake_args.append("-DCMAKE_CXX_COMPILER=clang++")
        add_env_cmake_setting(cmake_args, "CMAKE_LINKER_TYPE", default_value="LLD")

    add_env_cmake_setting(cmake_args, "SHORTFIN_ENABLE_LTO", default_value="ON")
    add_env_cmake_setting(cmake_args, "SHORTFIN_IREE_SOURCE_DIR")
    add_env_cmake_setting(cmake_args, "SHORTFIN_ENABLE_ASAN")
    add_env_cmake_setting(cmake_args, "SHORTFIN_ENABLE_TOKENIZERS", default_value="OFF")
    add_env_cmake_setting(cmake_args, "IREE_HIP_TEST_TARGET_CHIP", default_value="")

    # Only do a from-scratch configure if not already configured.
    cmake_cache_file = os.path.join(CMAKE_BUILD_DIR, "CMakeCache.txt")
    if not os.path.exists(cmake_cache_file):
        print(f"Configuring with: {cmake_args}")
        subprocess.check_call([CMAKE_EXE, SOURCE_DIR] + cmake_args, cwd=CMAKE_BUILD_DIR)
        print(f"CMake configure complete.")
    else:
        print(f"Not re-configing (already configured)")

    # Build.
    subprocess.check_call([CMAKE_EXE, "--build", "."], cwd=CMAKE_BUILD_DIR)
    print("Build complete.")

    # Optionally run CTests.
    if get_env_boolean("SHORTFIN_RUN_CTESTS", False):
        print("Running ctests...")
        subprocess.check_call(
            ["ctest", "--timeout", "30", "--output-on-failure"],
            cwd=CMAKE_BUILD_DIR,
        )


class CMakeBuildPy(_build_py):
    def run(self):
        # The super-class handles the pure python build.
        super().run()

        # Only build using cmake if not in prebuild mode.
        if is_cpp_prebuilt():
            return

        try:
            self.build_default_configuration()
            if ENABLE_TRACY:
                self.build_tracy_configuration()
        except subprocess.CalledProcessError as e:
            print("Native build failed:")
            traceback.print_exc()
            # This is not great, but setuptools *swallows* exceptions from here
            # and mis-reports them as deprecation warnings! This is fairly
            # fatal, so just kill it.
            sys.exit(1)

    def build_default_configuration(self):
        print("  *********************************")
        print("  * Building base shortfin        *")
        print("  *********************************")

        build_cmake_configuration(CMAKE_DEFAULT_BUILD_DIR)

        # Copy non-python binaries generated during the build.
        target_dir = os.path.join(os.path.abspath(self.build_lib), "_shortfin_default")

        print(f"Building in target: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        print("Copying build to target.")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(
            os.path.join(
                CMAKE_DEFAULT_BUILD_DIR,
                "python",
                "_shortfin_default",
            ),
            target_dir,
            symlinks=False,
        )

    def build_tracy_configuration(self):
        print("  *********************************")
        print("  * Building tracy shortfin       *")
        print("  *********************************")

        build_cmake_configuration(
            CMAKE_TRACY_BUILD_DIR, ["-DSHORTFIN_ENABLE_TRACING=ON"]
        )

        # Copy non-python binaries generated during the build.
        target_dir = os.path.join(os.path.abspath(self.build_lib), "_shortfin_tracy")

        print(f"Building in target: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        print("Copying build to target.")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(
            os.path.join(
                CMAKE_TRACY_BUILD_DIR,
                "python",
                "_shortfin_tracy",
            ),
            target_dir,
            symlinks=False,
        )


PYTHON_SOURCE_DIR = REL_SOURCE_DIR / "python"
PYTHON_DEFAULT_BINARY_DIR = REL_CMAKE_DEFAULT_BUILD_DIR / "python"
PYTHON_TRACY_BINARY_DIR = REL_CMAKE_TRACY_BUILD_DIR / "python"


# We need some directories to exist before setup.
def populate_built_package(abs_dir):
    """Makes sure that a directory and __init__.py exist.

    This needs to unfortunately happen before any of the build process
    takes place so that setuptools can plan what needs to be built.
    We do this for any built packages (vs pure source packages).
    """
    os.makedirs(abs_dir, exist_ok=True)
    with open(os.path.join(abs_dir, "__init__.py"), "wt"):
        pass


populate_built_package(os.path.join(PYTHON_DEFAULT_BINARY_DIR / "_shortfin_default"))
if ENABLE_TRACY:
    populate_built_package(os.path.join(PYTHON_TRACY_BINARY_DIR / "_shortfin_tracy"))

packages = find_namespace_packages(
    where=os.path.join(SOURCE_DIR, "python"),
    include=[
        "_shortfin",
        "_shortfin_default",
        "shortfin",
        "shortfin.*",
        "shortfin_apps",
        "shortfin_apps.*",
    ]
    + (["_shortfin_tracy"] if ENABLE_TRACY else []),
)
print(f"Found shortfin packages: {packages}")

setup(
    version=f"{PACKAGE_VERSION}",
    packages=packages,
    zip_safe=False,
    package_dir=combine_dicts(
        {
            "_shortfin": str(PYTHON_SOURCE_DIR / "_shortfin"),
            "_shortfin_default": str(PYTHON_DEFAULT_BINARY_DIR / "_shortfin_default"),
            "shortfin": str(PYTHON_SOURCE_DIR / "shortfin"),
            "shortfin_apps": str(PYTHON_SOURCE_DIR / "shortfin_apps"),
        },
        (
            ({"_shortfin_tracy": str(PYTHON_TRACY_BINARY_DIR / "_shortfin_tracy")})
            if ENABLE_TRACY
            else {}
        ),
    ),
    ext_modules=(
        [CMakeExtension("_shortfin_default.lib")]
        + ([CMakeExtension("_shortfin_tracy.lib")] if ENABLE_TRACY else [])
    ),
    cmdclass={
        "build": CustomBuild,
        "build_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
    },
)
