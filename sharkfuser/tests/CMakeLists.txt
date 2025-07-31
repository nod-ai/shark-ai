# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Find prebuilt Catch2 library
find_package(Catch2 3 REQUIRED)

# Find lit program
sharkfuser_find_program(lit "Please install lit (e.g., pip install lit).")

# Find FileCheck program
sharkfuser_find_program(FileCheck "Please install FileCheck (or bring it in via IREE distribution).")

# Find iree-opt program
sharkfuser_find_program(iree-opt "Please install IREE distribution.")


add_sharkfuser_test(
  NAME sharkfuser_attribute_tests
  SRCS
    test_attributes.cpp
    test_tensor_attributes.cpp
    test_conv_attributes.cpp
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
    test_ssa_validation.cpp
)

add_sharkfuser_test(
  NAME sharkfuser_logging_tests
  SRCS
    test_logging.cpp
)

add_sharkfuser_test(
  NAME sharkfuser_extras_tests
  SRCS
    test_extras.cpp
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
