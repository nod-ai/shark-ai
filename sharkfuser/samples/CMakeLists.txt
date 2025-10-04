# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_fusilli_sample(
  NAME fusilli_convolution_samples
  SRCS
    convolution/conv_fprop_nchw_kcrs.cpp
    convolution/conv_fprop_nhwc_krsc.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_sample(
  NAME fusilli_runtime_capi_samples
  SRCS
    runtime/runtime_api.c
  DEPS
    iree_runtime_unified
)
