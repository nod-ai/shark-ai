# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.28)

project(fusilli-plugin
  VERSION 0.1.0
  DESCRIPTION "Fusilli-Plugin: A Fusilli/IREE powered hipDNN plugin for graph JIT compilation."
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Local Includes
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(FusilliPluginDependencyUtils)

# Options
option(FUSILLI_PLUGIN_USE_LOCAL_FUSILLI "Use local Fusilli build from ../sharkfuser" ON)

# Dependencies
set(HIP_PLATFORM "amd")
find_package(hip REQUIRED)

fusilli_plugin_dependency(GTest GTEST_VERSION 1.16.0)
fusilli_plugin_dependency(IREERuntime)
fusilli_plugin_dependency(hipdnn_frontend HIP_DNN_HASH 8a49655a7cc73f1c766256a7ead3a987e830efe9)
fusilli_plugin_dependency(Fusilli USE_LOCAL ${FUSILLI_PLUGIN_USE_LOCAL_FUSILLI})

# Plugin definition
add_library(iree_plugin SHARED
    IreePlugin.cpp
)
target_link_libraries(iree_plugin PUBLIC hipdnn_sdk hip::host)
target_link_libraries(iree_plugin PRIVATE hip::host fusilli::fusilli)
target_compile_definitions(iree_plugin PRIVATE
    COMPONENT_NAME="iree_plugin"
)

# Tests
enable_testing()

set(IREE_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "iree plugin dir" FORCE)
add_subdirectory(integration_tests)
