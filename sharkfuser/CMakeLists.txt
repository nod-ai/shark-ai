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
  DESCRIPTION "Fusilli: A C++ Graph API and Frontend backed by IREE code-generated kernels"
  LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# https://discourse.cmake.org/t/cmake-3-28-cmake-cxx-compiler-clang-scan-deps-notfound-not-found/9244/3
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Add include directory for header-only library
add_library(libfusilli INTERFACE)
target_include_directories(libfusilli INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(libfusilli INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/include)

# Includes
include(CTest)

# Local Includes
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(CTestMacros)
include(SharkfuserUtils)

# Find IREE packages + tools
# It is up to the consumer of Fusilli to determine how to bring the IREE dependencies in,
# but here's our recommendation:
# - Get CLI tools (iree-compile, iree-opt) from python packages
# - For C-API interfacing with IREERuntime, have it built from source and statically linked
#
# Once IREERuntime is built, just point to the build location with -DIREERuntime_DIR or
# -DCMAKE_PREFIX_PATH, like so:
#
#     -DIREERuntime_DIR=</path/to/iree/build/lib/cmake/IREE>
#
# The dependency on IREECompiler is through the CLI tool (`iree-compile`) which is easily
# accessible from binary releases so a source build is not necessary here.
sharkfuser_find_program(iree-compile "Please install IREE compiler binary (e.g., pip install iree-base-compiler).")
find_package(IREERuntime REQUIRED)

# Transitive deps for IREERuntime
find_package(Threads REQUIRED)

# Create external_tools.h.inc
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/templates/external_tools.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/fusilli/support/external_tools.h.inc
  @ONLY
)

# Build options
option(SHARKFUSER_BUILD_TESTS "Builds C++ tests" ON)
option(SHARKFUSER_BUILD_SAMPLES "Builds C++ samples" ON)
option(SHARKFUSER_DEBUG_BUILD "Enable debug build options" OFF)
option(SHARKFUSER_CODE_COVERAGE "Enable code coverage for tests" OFF)

if(SHARKFUSER_DEBUG_BUILD OR SHARKFUSER_CODE_COVERAGE)
  message(STATUS "Setting CMAKE_BUILD_TYPE to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Debug build" FORCE)
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
