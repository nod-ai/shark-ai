# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(CONV_SAMPLES
  convolution/conv_fprop_nchw_kcrs.cpp
  convolution/conv_fprop_nhwc_krsc.cpp
  convolution/conv_fprop_nhwc_krsc_with_pad.cpp
  convolution/conv_fprop_with_bias.cpp
  convolution/conv_fprop_with_relu.cpp
  convolution/conv_fprop_with_relu_and_bias.cpp
)

foreach(file ${CONV_SAMPLES})
  # Extract the base name from the file path for unique naming
  get_filename_component(base_name ${file} NAME_WE)
  set(sample_name "fusilli_convolution_samples_${base_name}")

  add_fusilli_sample(
    NAME ${sample_name}
    SRCS
      ${file}
    DEPS
      libfusilli
      libutils
      Catch2::Catch2WithMain
  )
endforeach()

add_fusilli_sample(
  NAME fusilli_runtime_capi_samples
  SRCS
    runtime/runtime_api.c
  DEPS
    iree_runtime_unified
)
