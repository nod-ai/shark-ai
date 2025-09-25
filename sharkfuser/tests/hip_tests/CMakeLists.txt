# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

enable_language(HIP)  # enables HIP language for this dir and this dir only

add_fusilli_test(
  NAME fusilli_hip_tests
  SRCS
    test_hip_buffer_import_api.cpp
  DEPS
    iree_runtime_unified
    Catch2::Catch2WithMain
)

set_source_files_properties(test_hip_buffer_import_api.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(fusilli_hip_tests PROPERTIES CMAKE_HIP_ARCHITECTURES "gfx942")
target_compile_features(fusilli_hip_tests PRIVATE cxx_std_20)
