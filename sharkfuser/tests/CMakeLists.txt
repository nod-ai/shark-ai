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
