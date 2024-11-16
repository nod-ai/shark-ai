#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the `X.Y.Z[.dev]` version identifier from the
# 'sharktank' and 'shortfin' version files and computes the version
# for the meta 'shark-ai' package.
#
# Usage:
#   ./compute_common_version.py --stable-release --write-json
#   cat ../../shark-ai/version_local.json

import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("--write-json", action="store_true")
parser.add_argument("--version-suffix", action="store", type=str)

release_type = parser.add_mutually_exclusive_group()
release_type.add_argument("-stable", "--stable-release", action="store_true")  # default
release_type.add_argument("-rc", "--nightly-release", action="store_true")


args = parser.parse_args()

if not (args.stable_release or args.nightly_release):
    parser.print_usage(sys.stderr)
    sys.stderr.write("error: A release type is required\n")
    sys.exit(1)

if args.stable_release and args.version_suffix:
    sys.stderr.write("error: A version suffix is only supported for stable releases\n")
    sys.exit(1)

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent

VERSION_FILE_SHARKTANK = REPO_ROOT / "sharktank/version.json"
VERSION_FILE_SHORTFIN = REPO_ROOT / "shortfin/version.json"
VERSION_FILE_LOCAL = REPO_ROOT / "shark-ai/version_local.json"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def write_version_info():
    with open(VERSION_FILE_LOCAL, "w") as f:
        json.dump(version_local, f, indent=2)
        f.write("\n")


sharktank_version = load_version_info(VERSION_FILE_SHARKTANK)
SHARKTANK_PACKAGE_VERSION = sharktank_version.get("package-version")
SHARKTANK_BASE_VERSION = Version(SHARKTANK_PACKAGE_VERSION).base_version

shortfin_version = load_version_info(VERSION_FILE_SHORTFIN)
SHORTFIN_PACKAGE_VERSION = shortfin_version.get("package-version")
SHORTFIN_BASE_VERSION = Version(SHORTFIN_PACKAGE_VERSION).base_version

if SHARKTANK_BASE_VERSION > SHORTFIN_BASE_VERSION:
    COMMON_VERSION = SHARKTANK_BASE_VERSION
else:
    COMMON_VERSION = SHORTFIN_BASE_VERSION

if args.nightly_release:
    if args.version_suffix:
        VERSION_SUFFIX = args.version_suffix
    else:
        VERSION_SUFFIX = "rc" + datetime.today().strftime("%Y%m%d")

    COMMON_VERSION += VERSION_SUFFIX

if args.write_json:
    version_local = {"package-version": COMMON_VERSION}
    write_version_info()

print(COMMON_VERSION)
