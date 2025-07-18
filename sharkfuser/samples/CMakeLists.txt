# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Download Catch2 for testing
if(NOT catch2_FOUND)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY  https://github.com/catchorg/Catch2.git
    GIT_TAG         ${SHARKFUSER_CATCH2_GIT_TAG}
  )
  FetchContent_MakeAvailable(catch2)
endif()

add_sharkfuser_sample(
  NAME sharkfuser_convolution_samples
  SRCS
    convolution/conv_fprop.cpp
)
