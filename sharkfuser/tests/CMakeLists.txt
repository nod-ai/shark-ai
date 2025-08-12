# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Find prebuilt Catch2 library
find_package(Catch2 3 REQUIRED)

# Find lit program
sharkfuser_find_program(lit "Please install lit (e.g., pip install lit).")

# Find filecheck program
sharkfuser_find_program(filecheck "Please install filecheck (e.g., pip install filecheck).")

# Find iree-opt program
sharkfuser_find_program(iree-opt "Please install iree-opt (e.g., pip install iree-base-compiler).")

add_sharkfuser_test(
  NAME sharkfuser_attribute_tests
  SRCS
    test_attributes.cpp
    test_tensor_attributes.cpp
    test_conv_attributes.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_sharkfuser_test(
  NAME sharkfuser_node_tests
  SRCS
    test_conv_node.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_sharkfuser_test(
  NAME sharkfuser_graph_tests
  SRCS
    test_graph.cpp
    test_context.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_sharkfuser_test(
  NAME sharkfuser_support_tests
  SRCS
    test_cache.cpp
    test_ssa_validation.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

# Keep test_logging.cpp a separate test target as we manipulate
# logging env vars here which can suppress logs from other tests.
add_sharkfuser_test(
  NAME sharkfuser_logging_tests
  SRCS
    test_logging.cpp
  DEPS
    libfusilli
    Catch2::Catch2WithMain
)

add_sharkfuser_lit_test(
  SRC
    lit/test_asm_emitter.cpp
  DEPS
    libfusilli
  TOOLS
    filecheck
)

add_sharkfuser_lit_test(
  SRC
    lit/test_conv_asm_emitter.cpp
  DEPS
    libfusilli
  TOOLS
    filecheck
    iree-opt
)
