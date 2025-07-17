# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.28)

# Avoid in-source builds (prevent build artifacts from cluttering the source directory)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(
    FATAL_ERROR
      "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source."
  )
endif()

# Get version number from file
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.json VERSION_JSON_STRING)
string(JSON PACKAGE_VERSION GET ${VERSION_JSON_STRING} package-version)
string(REGEX MATCH "(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*" BASE_VERSION ${PACKAGE_VERSION})

project(
  sharkfuser
  VERSION ${BASE_VERSION}
  DESCRIPTION "Fusili: A C++ Graph API and Frontend backed by IREE code-generated kernels"
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# https://discourse.cmake.org/t/cmake-3-28-cmake-cxx-compiler-clang-scan-deps-notfound-not-found/9244/3
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Add include directory for header-only library
add_library(libfusili INTERFACE)
target_include_directories(libfusili INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Version pins for dependencies
set(SHARKFUSER_IREE_GIT_TAG "iree-3.6.0rc20250612")
set(SHARKFUSER_CATCH2_GIT_TAG "v3.8.1")

# Includes
include(FetchContent)
include(CTest)

# Local Includes
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(CTestMacros)

# Build options
option(SHARKFUSER_BUILD_TESTS "Builds C++ tests" ON)
option(SHARKFUSER_BUILD_SAMPLES "Builds C++ samples" ON)
option(SHARKFUSER_SYSTEMS_AMDGPU "Builds for AMD GPU systems" ON)
option(SHARKFUSER_DEBUG_BUILD "Enable debug build options" OFF)
option(SHARKFUSER_CODE_COVERAGE "Enable code coverage for tests" OFF)

set(SHARKFUSER_IREE_SOURCE_DIR "" CACHE FILEPATH "Path to local IREE source directory")

if(SHARKFUSER_DEBUG_BUILD OR SHARKFUSER_CODE_COVERAGE)
  message(STATUS "Setting CMAKE_BUILD_TYPE to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Debug build" FORCE)
endif()

message(STATUS "SharkFuser supported systems:")
if(SHARKFUSER_SYSTEMS_AMDGPU)
  message(STATUS "  - AMD GPU")
endif()
message(STATUS "  - Host")


################################################################################
# Bring in IREE Dependency
################################################################################

set(IREE_VISIBILITY_HIDDEN OFF)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC OFF)
set(IREE_HAL_DRIVER_LOCAL_TASK OFF)
set(IREE_HAL_DRIVER_VULKAN OFF)
if(SHARKFUSER_SYSTEMS_AMDGPU)
  set(IREE_HAL_DRIVER_HIP ON)
endif()

if(SHARKFUSER_IREE_SOURCE_DIR)
  message(STATUS "Using local IREE source directory: ${SHARKFUSER_IREE_SOURCE_DIR}")
  add_subdirectory(${SHARKFUSER_IREE_SOURCE_DIR} sharkfuser_iree SYSTEM EXCLUDE_FROM_ALL)
else()
  message(STATUS "Fetching IREE sources from tag ${SHARKFUSER_IREE_GIT_TAG}")
  set(IREE_SUBMODULES "third_party/benchmark third_party/cpuinfo third_party/flatcc third_party/hip-build-deps")
  FetchContent_Declare(
    sharkfuser_iree
    GIT_REPOSITORY  https://github.com/iree-org/iree.git
    GIT_TAG         ${SHARKFUSER_IREE_GIT_TAG}
    GIT_SUBMODULES  ${IREE_SUBMODULES}
    GIT_SHALLOW     TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(sharkfuser_iree)
  if(NOT sharkfuser_iree_POPULATED)
    FetchContent_MakeAvailable(sharkfuser_iree)
  endif()
endif()

# Build samples
if(SHARKFUSER_BUILD_SAMPLES)
  message(STATUS "Building SharkFuser samples")
  add_subdirectory(samples)
endif()

# Build tests
if(SHARKFUSER_BUILD_TESTS)
  message(STATUS "Building SharkFuser tests")
  add_subdirectory(tests)
  enable_testing()
endif()

# Code coverage exclusions
configure_file(
  ${CMAKE_SOURCE_DIR}/build_tools/cmake/CTestCustom.cmake.in
  ${CMAKE_BINARY_DIR}/CTestCustom.cmake
  COPYONLY
)
