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
  fusilli
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

# Includes
include(CTest)

# Local Includes
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(FusilliBuildUtils)
include(FusilliTestUtils)

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
fusilli_find_program(iree-compile "Please install IREE compiler binary (e.g., pip install iree-base-compiler).")
find_package(IREERuntime REQUIRED)

# Transitive deps for IREERuntime
find_package(Threads REQUIRED)

# Defines variables CMAKE_INSTALL_<dir> for common install locations.
include(GNUInstallDirs)

# Create external_tools.h.inc
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/templates/external_tools.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/fusilli/support/external_tools.h.inc
  @ONLY
)

# Create header-only library.
add_library(libfusilli INTERFACE)

# Consumers of exported/installed target use fusilli::fusilli.
set_target_properties(libfusilli PROPERTIES EXPORT_NAME fusilli)

# Create alias for consistent naming in consuming projects building locally (via
# FetchContent) and against an installed version (via find_package).
add_library(fusilli::fusilli ALIAS libfusilli)

# INTERFACE libraries don't install/export header files added via
# `target_include_directories`. Using a file set property ensures headers are
# available in the installed/exported package.
file(GLOB_RECURSE FUSILLI_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
  ${CMAKE_CURRENT_BINARY_DIR}/include/*.h.inc
)
target_sources(libfusilli INTERFACE
  FILE_SET fusilli_headers TYPE HEADERS # file sets require lowercase name
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
  FILES ${FUSILLI_HEADERS}
)

# Statically link IREERuntime
target_link_libraries(libfusilli INTERFACE iree_runtime_unified)

# Build options
option(FUSILLI_BUILD_TESTS "Builds C++ tests and samples" ON)
option(FUSILLI_BUILD_BENCHMARKS "Builds C++ benchmarks" ON)
option(FUSILLI_CODE_COVERAGE "Enable code coverage for tests" OFF)
option(FUSILLI_ENABLE_LOGGING "Enable logging for tests and samples" OFF)

# Build Type - Release/Debug
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting CMAKE_BUILD_TYPE to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release build" FORCE)
endif()
if(FUSILLI_CODE_COVERAGE)
  message(STATUS "Setting CMAKE_BUILD_TYPE to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Debug build" FORCE)
endif()

# Option for AMDGPU backend
option(FUSILLI_SYSTEMS_AMDGPU "Builds for AMD GPU systems" ON)
message(STATUS "Fusilli supported systems:")
if(FUSILLI_SYSTEMS_AMDGPU)
  message(STATUS "  - AMD GPU")
endif()
message(STATUS "  - Host")

# Enable AMD GPU samples and tests if GPU is detected
if(FUSILLI_SYSTEMS_AMDGPU)
  # Check for Linux AMD GPU device node - `/dev/kfd`
  if(EXISTS "/dev/kfd")
    message(STATUS "AMD GPU detected: Enabling GPU samples and tests")
    add_definitions(-DFUSILLI_ENABLE_AMDGPU)
  else()
    message(FATAL_ERROR "FUSILLI_SYSTEMS_AMDGPU is ON but `/dev/kfd` not found: No AMD GPU detected on this system.")
  endif()
endif()

# Both tests and benchmarks depend on libutils
if(FUSILLI_BUILD_TESTS OR FUSILLI_BUILD_BENCHMARKS)
  # Add library for tests/utils.h
  add_library(libutils INTERFACE)
  target_include_directories(libutils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

# Build tests and samples
if(FUSILLI_BUILD_TESTS)
  message(STATUS "Building Fusilli tests and samples")

  # Find prebuilt Catch2 library
  find_package(Catch2 3 REQUIRED)

  # Add tests and samples sub directories
  add_subdirectory(tests)
  add_subdirectory(samples)

  # Activate CTest integration
  enable_testing()
endif()

# Build benchmarks
if(FUSILLI_BUILD_BENCHMARKS)
  message(STATUS "Building Fusilli benchmarks")

  # Get CLI11 (argparse library for C++)
  include(FetchContent)
  FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
  )
  FetchContent_MakeAvailable(cli11_proj)

  # Add benchmarks sub directory
  add_subdirectory(benchmarks)
endif()

# Add libfusilli target to export set
install(TARGETS libfusilli
  EXPORT FusilliTargets
  FILE_SET fusilli_headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create FusilliTargets.cmake in the install tree from FusilliTargets export set
install(EXPORT FusilliTargets
  FILE FusilliTargets.cmake
  NAMESPACE fusilli::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)

# Create FusilliTargets.cmake in the build tree from FusilliTargets export set,
# allowing for other projects to find_package directly from the build directory
# without installing.
export(TARGETS libfusilli
  FILE ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/FusilliTargets.cmake
  NAMESPACE fusilli::
)

# Generate config file in build directory.
#
# configure_package_config_file uses INSTALL_DESTINATION to calculate relative
# paths within the generated config file. But it does NOT actually install the
# file, you must use install(FILES...) for that. INSTALL_DESTINATION must match
# the DESTINATION in the install(FILES...) command below, but CMake doesn't
# verify that for you.
#
# We're generating FusilliConfig.cmake in the build tree, so no extra work is
# needed to make it consumable via find_package directly from the build
# directory, i.e. we don't need an `export(FILE...` call.
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/build_tools/cmake/templates/FusilliConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/FusilliConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)

# Write FusilliConfig.cmake file to install tree
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/FusilliConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)
