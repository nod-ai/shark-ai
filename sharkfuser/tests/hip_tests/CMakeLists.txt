# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Hip dependency for targets in this dir and this dir only
enable_language(HIP)

# clangd doesn't handle hip device side code well currently, separate device
# code into separate library to limit broken intellisense.
set_source_files_properties(hip_kernels.cpp PROPERTIES LANGUAGE HIP)
add_library(hip_kernels
    hip_kernels.cpp
)

add_fusilli_tests(
  PREFIX fusilli_hip_tests
  SRCS
    test_hip_buffer_import_api.cpp
  DEPS
    hip::host
    hip_kernels
    iree_runtime_unified
    Catch2::Catch2WithMain
)
