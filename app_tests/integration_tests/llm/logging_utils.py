# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os


def start_log_group(headline):
    """Start a collapsible log group in GitHub Actions."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return f"\n::group::{headline}"
    return ""


def end_log_group():
    """End a collapsible log group in GitHub Actions."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "\n::endgroup::"
    return ""
