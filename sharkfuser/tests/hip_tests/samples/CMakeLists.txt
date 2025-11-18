# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_fusilli_samples(
  PREFIX fusilli_hip_tests_conv_samples
  SRCS
    conv_fprop_nchw_kcrs.cpp
  DEPS
    hip::host
    libfusilli
    libutils
    libhiputils
    Catch2::Catch2WithMain
)
