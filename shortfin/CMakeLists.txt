# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.29)

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
  "libshortfin"
  VERSION ${BASE_VERSION}
  LANGUAGES C CXX)

include(CMakeDependentOption)

set(SOVERSION 1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
# https://discourse.cmake.org/t/cmake-3-28-cmake-cxx-compiler-clang-scan-deps-notfound-not-found/9244/3
set(CMAKE_CXX_SCAN_FOR_MODULES 0)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Problems with linking libfmt without PIC.
# Turn on PIC on non windows targets.
if(NOT WIN32)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Pins
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/build_tools/iree_version.txt IREE_VERSION_STRING)
set(SHORTFIN_IREE_GIT_TAG "iree-${IREE_VERSION_STRING}")

# build options
option(SHORTFIN_BUILD_PYTHON_BINDINGS "Builds Python Bindings" OFF)
option(SHORTFIN_BUILD_TESTS "Builds C++ tests" ON)
option(SHORTFIN_BUNDLE_DEPS "Download dependencies instead of using system libraries" ON)
option(SHORTFIN_ENABLE_TRACING "Enable runtime tracing for iree and shortfin" OFF)
option(SHORTFIN_ENABLE_LTO "Enables LTO if supported" ON)

set(SHORTFIN_IREE_SOURCE_DIR "" CACHE FILEPATH "Path to IREE source")

# Options for building static or dynamic libraries.
# Default to dynamic linking, unless on Windows.
# TODO(#211): Unify the defaults once Windows dynamic linking issues are fixed.
set(SHORTFIN_BUILD_STATIC_DEFAULT OFF)
set(SHORTFIN_BUILD_DYNAMIC_DEFAULT ON)
if(WIN32)
  set(SHORTFIN_BUILD_STATIC_DEFAULT ON)
  set(SHORTFIN_BUILD_DYNAMIC_DEFAULT OFF)
endif()
option(SHORTFIN_BUILD_STATIC "Builds static libraries" ${SHORTFIN_BUILD_STATIC_DEFAULT})
option(SHORTFIN_BUILD_DYNAMIC "Builds dynamic libraries" ${SHORTFIN_BUILD_DYNAMIC_DEFAULT})
cmake_dependent_option(SHORTFIN_LINK_DYNAMIC "Links internal binaries against static libshortfin.a" ON "SHORTFIN_BUILD_DYNAMIC" OFF)
if(NOT SHORTFIN_BUILD_STATIC AND NOT SHORTFIN_BUILD_DYNAMIC)
  message(FATAL_ERROR "One of SHORTFIN_BUILD_STATIC or SHORTFIN_BUILD_DYNAMIC must be ON")
endif()
message(STATUS "Shortfin build static = ${SHORTFIN_BUILD_STATIC}, dynamic = ${SHORTFIN_BUILD_DYNAMIC}")
if(SHORTFIN_LINK_DYNAMIC)
  message(STATUS "Dynamic linking to shortfin")
  set(SHORTFIN_LINK_LIBRARY_NAME "shortfin")
else()
  message(STATUS "Static linking to shortfin-static")
  set(SHORTFIN_LINK_LIBRARY_NAME "shortfin-static")
endif()

# Includes.
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(shortfin_library)
include(CheckCXXCompilerFlag)
include(FetchContent)

################################################################################
# Toolchain features
################################################################################

if(SHORTFIN_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT SHORTFIN_LTO_SUPPORTED OUTPUT SHORTFIN_LTO_ERROR)
  if(SHORTFIN_LTO_SUPPORTED)
    message(STATUS "Shortfin LTO Enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "Could not enable LTO (not supported): ${SHORTFIN_LTO_ERROR}")
  endif()
endif()

# Enabling ASAN. Note that this will work best if building in a completely
# bundled fashion and with an ASAN rigged CPython. Otherwise, various LD_PRELOAD
# hacks are needed. This is merely a develope convenience: people are more
# than welcome to set flags themselves.
option(SHORTFIN_ENABLE_ASAN "Enable ASAN" OFF)
if(SHORTFIN_ENABLE_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)

  # Enable more ASAN checks.
  add_compile_definitions(IREE_SANITIZER_ADDRESS)
endif()

# Thread safety annotations: Enabled if the compiler supports it.
check_cxx_compiler_flag("-Wthread-safety" SHORTFIN_HAS_THREAD_SAFETY_ANNOTATIONS)
if(SHORTFIN_HAS_THREAD_SAFETY)
  add_compile_options(-Wthread-safety)
  add_compile_definitions(SHORTFIN_HAS_THREAD_SAFETY_ANNOTATIONS)
endif()

option(SHORTFIN_SYSTEMS_AMDGPU "Builds for AMD GPU systems" ON)
message(STATUS "shortfin supported systems:")
if(SHORTFIN_SYSTEMS_AMDGPU)
  message(STATUS "  - AMD GPU")
endif()
message(STATUS "  - Host")

################################################################################
# Dependencies
################################################################################

if(SHORTFIN_BUNDLE_DEPS)
  ## fmt
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1 (sync with spdlog)
  )

  ## spdlog
  # We build fmt from source instead, because we also use fmt.
  set(SPDLOG_FMT_EXTERNAL ON)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        2d4acf8cc321d7783d8f2e22e17a794c6d0e9450 # v1.14.1
  )

  ## xtl: required for xtensor
  FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        a7c1c5444dfc57f76620391af4c94785ff82c8d6 # v0.7.7
  )

  ## xtensor
  FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        3634f2ded19e0cf38208c8b86cea9e1d7c8e397d # v0.25.0
  )

  # In order to bundle libraries without conflict, we have to tweak settings.
  shortfin_push_bundled_lib_options()
  # Enable spdlog shared library options so we can export it.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_SHARED_LIB -Dspdlog_EXPORTS")
  FetchContent_MakeAvailable(fmt spdlog xtl xtensor)
  shortfin_pop_bundled_lib_options()
else()
  find_package(spdlog)
  find_package(xtensor)
endif()

################################################################################
# IREE
################################################################################

# Set IREE build flags.
# We currently rely on IREE to have visible symbols in order to re-export
# its API for further use.
# TODO: Turn this back on and use explicit visibility control in the IREE
# runtime and linker scripts.
set(IREE_VISIBILITY_HIDDEN OFF)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
# Disable missing submodules error because we are only building the runtime.
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
# Only enable local_sync/local_task/hip drivers for now.
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
set(IREE_HAL_DRIVER_LOCAL_TASK ON)
if(SHORTFIN_SYSTEMS_AMDGPU)
  set(IREE_HAL_DRIVER_HIP ON)
endif()
if (SHORTFIN_ENABLE_TRACING)
  set(IREE_ENABLE_RUNTIME_TRACING ON)
  # When using shared libraries there are some issues that need to be
  # explored more on static initialization order. Something is getting
  # initialized and is emitting tracy events before tracy objects are
  # initialized. This can point to some shared library overloading allocation
  # functions and making them emit tracy events, which are further used in
  # some static allocation. See https://github.com/wolfpld/tracy/issues/196
  # for a similar issue and discussion. Using the workaround suggested in
  # that issue for now. Note that this does not happen when using static
  # libraries.
  set(TRACY_DELAYED_INIT ON CACHE BOOL "Enable delayed init for tracy")
endif()

# In order to bundle libraries without conflict, we have to tweak settings.
shortfin_push_bundled_lib_options()
if(SHORTFIN_IREE_SOURCE_DIR)
  message(STATUS "Using existing IREE sources: ${SHORTFIN_IREE_SOURCE_DIR}")
  add_subdirectory(${SHORTFIN_IREE_SOURCE_DIR} shortfin_iree SYSTEM EXCLUDE_FROM_ALL)
else()
  message(STATUS "Fetching IREE sources from tag ${SHORTFIN_IREE_GIT_TAG}")

  # TODO: We shouldn't have to pull googletest when we are not building tests.
  #       This needs to be fixed with IREE.
  set(IREE_SUBMODULES "third_party/benchmark third_party/cpuinfo third_party/flatcc third_party/hip-build-deps third_party/googletest")
  if (SHORTFIN_ENABLE_TRACING)
    set(IREE_SUBMODULES "${IREE_SUBMODULES} third_party/tracy")
  endif()
  FetchContent_Declare(
    shortfin_iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG "${SHORTFIN_IREE_GIT_TAG}"
    GIT_SUBMODULES ${IREE_SUBMODULES}
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(shortfin_iree)
  if(NOT shortfin_iree_POPULATED)
    FetchContent_MakeAvailable(shortfin_iree)
  endif()
endif()
shortfin_pop_bundled_lib_options()

################################################################################
# Tests
################################################################################

if(SHORTFIN_BUILD_TESTS)
  if (NOT SHORTFIN_BUNDLE_DEPS AND NOT SHORTFIN_IREE_SOURCE_DIR)
    # For now we use gtest shipped alongside with IREE.
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)
  enable_testing()
endif()


add_subdirectory(src)

if(SHORTFIN_BUILD_PYTHON_BINDINGS)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  add_subdirectory(python)
  set(SHORTFIN_PYTHON_CPP_PREBUILT "TRUE") # See setup.py.
  configure_file(setup.py setup.py @ONLY)
  configure_file(pyproject.toml pyproject.toml COPYONLY)
endif()
