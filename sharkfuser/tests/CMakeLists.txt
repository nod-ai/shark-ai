# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Find prebuilt Catch2 library
find_package(Catch2 3 REQUIRED)

# Find lit program (pip or system installed)
sharkfuser_find_external_tool(lit "Please install lit (e.g., pip install lit).")

# Find FileCheck program + wrap in a CMake target interface
sharkfuser_find_external_tool(FileCheck "Please install FileCheck (or bring it in via IREE distribution).")
add_executable(FileCheck IMPORTED GLOBAL)
set_target_properties(FileCheck PROPERTIES IMPORTED_LOCATION "${SHARKFUSER_EXTERNAL_FILECHECK}")

# Find iree-opt program + wrap CMake target interface
sharkfuser_find_external_tool(iree-opt "Please install IREE.")
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
  SRC lit/test_asm_emitter.cpp
  TOOLS
    FileCheck
)

add_sharkfuser_lit_test(
  SRC lit/test_conv_asm_emitter.cpp
  TOOLS
    FileCheck
    iree-opt
)
