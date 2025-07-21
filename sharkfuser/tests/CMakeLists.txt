# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Find prebuilt Catch2 library
find_package(Catch2 3 REQUIRED)

# Find lit program (pip or system installed)
if(NOT SHARKFUSER_EXTERNAL_LIT)
  find_program(SHARKFUSER_EXTERNAL_LIT NAMES lit)
  if(NOT SHARKFUSER_EXTERNAL_LIT)
    message(FATAL_ERROR "Could not find 'lit' in PATH. Please install lit (e.g., pip install lit).")
  endif()
endif()
message(STATUS "Using lit: ${SHARKFUSER_EXTERNAL_LIT}")

# Find filecheck program (pip or system installed)
if(NOT SHARKFUSER_EXTERNAL_FILECHECK)
  find_program(SHARKFUSER_EXTERNAL_FILECHECK NAMES filecheck)
  if(NOT SHARKFUSER_EXTERNAL_FILECHECK)
    message(FATAL_ERROR "Could not find 'filecheck' in PATH. Please install filecheck (e.g., pip install filecheck).")
  endif()
endif()
message(STATUS "Using filecheck: ${SHARKFUSER_EXTERNAL_FILECHECK}")


add_sharkfuser_test(
  NAME sharkfuser_attribute_tests
  SRCS
    test_attributes.cpp
    test_tensor_attributes.cpp
    test_conv_attributes.cpp
)

add_sharkfuser_test(
  NAME sharkfuser_tensor_tests
  SRCS
    test_tensor.cpp
)

add_sharkfuser_test(
  NAME sharkfuser_node_tests
  SRCS
    test_conv_node.cpp
)

add_sharkfuser_test(
  NAME sharkfuser_graph_tests
  SRCS
    test_graph.cpp
    test_context.cpp
)

add_sharkfuser_test(
  NAME sharkfuser_logging_tests
  SRCS
    test_logging.cpp
)

add_sharkfuser_lit_test(
  SRC test_example_lit.cpp
)
