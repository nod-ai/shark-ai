# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(FUSILLI_CONV_SAMPLE_FILES
  convolution/conv_fprop_nchw_kcrs.cpp
  convolution/conv_fprop_nchw_kcrs_grouped.cpp
  convolution/conv_fprop_nhwc_krsc.cpp
  convolution/conv_fprop_nhwc_krsc_with_pad.cpp
  convolution/conv_fprop_with_bias.cpp
  convolution/conv_fprop_with_relu.cpp
  convolution/conv_fprop_with_relu_and_bias.cpp
  convolution/conv_fprop_grouped_with_relu_and_bias.cpp
  convolution/conv_wgrad_nhwc_krsc.cpp
  convolution/conv_wgrad_nhwc_krsc_grouped.cpp
  convolution/conv_dgrad_nhwc_krsc.cpp
  convolution/conv_dgrad_nhwc_krsc_grouped.cpp)

# TODO(#2630): Currently the test with grouped not unit-stride ConvWGrad has SIGSEGV on CPU backend
if(FUSILLI_SYSTEMS_AMDGPU)
  list(APPEND FUSILLI_CONV_SAMPLE_FILES convolution/conv_wgrad_nhwc_krsc_grouped_strided.cpp)
endif()

add_fusilli_samples(
  PREFIX fusilli_conv_samples
  SRCS
    ${FUSILLI_CONV_SAMPLE_FILES}
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)

add_fusilli_samples(
  PREFIX fusilli_pointwise_samples
  SRCS
    pointwise/pointwise_binary_ops.cpp
    pointwise/pointwise_unary_ops.cpp
  DEPS
    libfusilli
    libutils
    Catch2::Catch2WithMain
)
