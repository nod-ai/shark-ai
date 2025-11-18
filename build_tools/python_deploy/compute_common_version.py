#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the `X.Y.Z[.dev]` version identifier from the
# 'amdsharktank' and 'shortfin' version files and computes the version
# for the meta 'amdshark-ai' package.
#
# Usage:
#   ./compute_common_version.py --stable-release --write-json
#   cat ../../amdshark-ai/version_local.json

import argparse
from pathlib import Path
import json
from datetime import datetime
import subprocess

from packaging.version import Version


parser = argparse.ArgumentParser()
parser.add_argument("--write-json", action="store_true")

release_type = parser.add_mutually_exclusive_group(required=True)
release_type.add_argument("-stable", "--stable-release", action="store_true")
release_type.add_argument("-rc", "--nightly-release", action="store_true")
release_type.add_argument("-dev", "--development-release", action="store_true")
release_type.add_argument("--version-suffix", action="store", type=str)

args = parser.parse_args()

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent

VERSION_FILE_amdsharkTANK_PATH = REPO_ROOT / "amdsharktank/version.json"
VERSION_FILE_amdsharkTUNER_PATH = REPO_ROOT / "amdsharktuner/version.json"
VERSION_FILE_SHORTFIN_PATH = REPO_ROOT / "shortfin/version.json"
VERSION_FILE_LOCAL_PATH = REPO_ROOT / "amdshark-ai/version_local.json"


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def write_version_info(version_file, version):
    with open(version_file, "w") as f:
        json.dump({"package-version": version}, f, indent=2)
        f.write("\n")


amdsharktank_version = load_version_info(VERSION_FILE_amdsharkTANK_PATH)
amdsharktank_package_version = amdsharktank_version.get("package-version")
amdsharktank_base_version = Version(amdsharktank_package_version).base_version

amdsharktuner_version = load_version_info(VERSION_FILE_amdsharkTUNER_PATH)
amdsharktuner_package_version = amdsharktuner_version.get("package-version")
amdsharktuner_base_version = Version(amdsharktuner_package_version).base_version

shortfin_version = load_version_info(VERSION_FILE_SHORTFIN_PATH)
shortfin_package_version = shortfin_version.get("package-version")
shortfin_base_version = Version(shortfin_package_version).base_version

common_version = shortfin_base_version
if amdsharktank_base_version > common_version:
    common_version = amdsharktank_base_version

if amdsharktuner_base_version > common_version:
    common_version = amdsharktuner_base_version

if args.nightly_release:
    common_version += "rc" + datetime.today().strftime("%Y%m%d")
elif args.development_release:
    common_version += (
        ".dev0+"
        + subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
elif args.version_suffix:
    common_version += args.version_suffix

if args.write_json:
    write_version_info(VERSION_FILE_LOCAL_PATH, common_version)

print(common_version)
