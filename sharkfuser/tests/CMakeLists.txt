# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Find prebuilt Catch2 library
find_package(Catch2 3 REQUIRED)

# Find lit program (pip or system installed)
sharkfuser_find_external_tool(lit "Please install lit (e.g., pip install lit).")

# Find filecheck program (pip or system installed) + wrap in a CMake target interface
sharkfuser_find_external_tool(filecheck "Please install filecheck (e.g., pip install filecheck).")
add_executable(filecheck IMPORTED GLOBAL)
set_target_properties(filecheck PROPERTIES IMPORTED_LOCATION "${SHARKFUSER_EXTERNAL_FILECHECK}")

# Find iree-opt program (pip or system installed) + wrap CMake target interface
sharkfuser_find_external_tool(iree-opt "Please install filecheck (e.g., pip install iree-base-compiler).")
add_executable(iree-opt IMPORTED GLOBAL)
set_target_properties(iree-opt PROPERTIES IMPORTED_LOCATION "${SHARKFUSER_EXTERNAL_IREE_OPT}")


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
  TOOLS
    filecheck
    iree-opt
)
