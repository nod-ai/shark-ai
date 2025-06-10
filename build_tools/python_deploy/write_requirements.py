#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script writes the `packaging/shark-ai/requirements.txt` file and pins
# the versions of the dependencies accordingly. For nightly releases,
#  * sharktank
#  * shortfin
# get pinned to the corresponding nightly version. The IREE packages are
# unpinned. For stable releases,
# * iree-base-compiler
# * iree-base-runtime
# * iree-turbine
# * sharktank
# * shortfin
# * sharktuner
# get pinned to the corresponding `X.Y.*` version.

import argparse
from pathlib import Path
import json

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("--version-suffix", action="store", type=str)

args = parser.parse_args()


THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent

VERSION_FILE_LOCAL = REPO_ROOT / "shark-ai/version_local.json"
VERSION_FILE_SHARKTANK = REPO_ROOT / "sharktank/version_local.json"
VERSION_FILE_SHARKTUNER = REPO_ROOT / "sharktuner/version_local.json"
VERSION_FILE_SHORTFIN = REPO_ROOT / "shortfin/version_local.json"
REQUIREMENTS_TXT = REPO_ROOT / "shark-ai/requirements.txt"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def write_requirements(requirements):
    with open(REQUIREMENTS_TXT, "w") as f:
        f.write("%s\n" % requirements)


metapackage_version = load_version_info(VERSION_FILE_LOCAL)
PACKAGE_VERSION = metapackage_version.get("package-version")

sharktank_version = load_version_info(VERSION_FILE_SHARKTANK)
SHARKTANK_PACKAGE_VERSION = sharktank_version.get("package-version")

sharktuner_version = load_version_info(VERSION_FILE_SHARKTUNER)
SHARKTUNER_PACKAGE_VERSION = sharktuner_version.get("package-version")

shortfin_version = load_version_info(VERSION_FILE_SHORTFIN)
SHORTFIN_PACKAGE_VERSION = shortfin_version.get("package-version")

stable_packages_list = ["iree-base-compiler", "iree-base-runtime", "iree-turbine"]

if Version(PACKAGE_VERSION).is_prerelease:
    requirements = ""
    for package in stable_packages_list:
        requirements += package + "\n"
    requirements = (
        "sharktank=="
        + Version(SHARKTANK_PACKAGE_VERSION).base_version
        + args.version_suffix
        + "\n"
    )
    requirements += (
        "sharktuner=="
        + Version(SHARKTUNER_PACKAGE_VERSION).base_version
        + args.version_suffix
        + "\n"
    )
    requirements += (
        "shortfin=="
        + Version(SHORTFIN_PACKAGE_VERSION).base_version
        + args.version_suffix
    )

    write_requirements(requirements)

else:
    MAJOR_VERSION = Version(PACKAGE_VERSION).major
    MINOR_VERSION = Version(PACKAGE_VERSION).minor

    STABLE_VERSION_TO_PIN = str(MAJOR_VERSION) + "." + str(MINOR_VERSION) + ".*"

    requirements = ""
    for package in stable_packages_list:
        requirements += package + "==" + STABLE_VERSION_TO_PIN + "\n"
    requirements += (
        "sharktank==" + Version(SHARKTANK_PACKAGE_VERSION).base_version + "\n"
    )
    requirements += (
        "sharktuner==" + Version(SHARKTUNER_PACKAGE_VERSION).base_version + "\n"
    )
    requirements += "shortfin==" + Version(SHORTFIN_PACKAGE_VERSION).base_version

    write_requirements(requirements)
