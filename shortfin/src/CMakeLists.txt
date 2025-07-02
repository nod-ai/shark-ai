# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Any definitions which must be reflected on the public library must be added
# to this library.
add_library(shortfin_public_defs INTERFACE)

add_subdirectory(shortfin)

# Common definitions exported from both static and dynamic libraries.
add_library(_shortfin_defs INTERFACE)
target_include_directories(
  _shortfin_defs INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)


get_property(
  _SHORTFIN_LIB_OPTIONAL_COMPONENTS GLOBAL PROPERTY SHORTFIN_LIB_OPTIONAL_COMPONENTS)

message(STATUS "Linking optional components '${_SHORTFIN_LIB_OPTIONAL_COMPONENTS}'")
shortfin_public_library(
  NAME
    shortfin
  LINUX_LD_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/shortfin.ld
  COMPONENTS
    shortfin_array
    shortfin_llm_components
    shortfin_local
    shortfin_support
    shortfin_systems_factory
    ${_SHORTFIN_LIB_OPTIONAL_COMPONENTS}
  USAGE_DEPS
    shortfin_public_defs
    spdlog::spdlog
    fmt::fmt
    xtensor
    xtl
    iree_defs
)
