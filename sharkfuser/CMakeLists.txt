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

# Insure iree runtime is linked when using libfusilli.
target_link_libraries(libfusilli INTERFACE
  iree_runtime_unified
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

# Add libfusilli target to export set
install(TARGETS libfusilli
  EXPORT fusilliTargets
  FILE_SET fusilli_headers    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create fusilliTargets.cmake in the install tree from fusilliTargets export set
install(EXPORT fusilliTargets
  FILE fusilliTargets.cmake
  NAMESPACE fusilli::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)

# Create fusilliTargets.cmake in the build tree fusilliTargets export set,
# allowing for other projects find_package directly from the build directory
# without installing.
export(TARGETS libfusilli
  FILE ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/fusilliTargets.cmake
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
# We're generating fusilliConfig.cmake in the build tree, so no extra work is
# needed to make it consumable via find_package directly from the build
# directory, i.e. we don't need an `export(FILE...` call.
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/templates/fusilliConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/fusilliConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)

# Write Config.cmake file to install tree
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/fusilli/fusilliConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusilli
)
