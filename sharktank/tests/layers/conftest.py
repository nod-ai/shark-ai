# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

def pytest_addoption(parser):
    """Add command line option for IRPA file path."""
    parser.addoption(
        "--irpa-path",
        action="store",
        default=None,
        help="Path to the IRPA file for testing"
    )
