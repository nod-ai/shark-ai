# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#===----------------------------------------------------------------------===#
#
# Provides dependencies correctly configured for fusilli-plugin build.
#
# Main entry point:
#   fusilli_plugin_dependency(DEP_NAME [args...])
#
# `fusilli_plugin_dependency` routes to lower level `_fetch_X` macros to
# actually fetch dependency `X`. Each `_fetch_X` macro preferentially
# `find_package`s installed/system versions of packages and falls back to
# vendoring dependencies in the build tree with `FetchContent`.
#
# Supported dependencies: GTest, hipdnn_frontend, Fusilli, IREERuntime
#
#===----------------------------------------------------------------------===#

cmake_minimum_required(VERSION 3.25.2)

include(FetchContent)

# Provide a fusilli plugin dependency. `fusilli_plugin_dependency` will
# preferentially use system version (available through `find_package`) of a
# dependency, and fall back to building local copy with `FetchContent` +
# configuration.
#
#  fusilli_plugin_dependency(
#    DEP_NAME
#    [<dependency-specific args>...]
#  )
#
# DEP_NAME
#   Supported dependencies:
#     GTest
#     hipdnn_frontend
#     Fusilli
#     IREERuntime
#
# <dependency-specific args>
#   The `_fetch_X` macro for dependency X defines the available options.
#   Examples: GTEST_VERSION for GTest, HIP_DNN_HASH for hipdnn_frontend
function(fusilli_plugin_dependency dep_name)
    # Route to appropriate `_fetch_X` macro.
    if(COMMAND _fetch_${dep_name})
        _with_logging(_fetch_${dep_name} ${dep_name} ${ARGN})
    else()
        message(FATAL_ERROR "Unknown dependency: ${dep_name}")
    endif()
endfunction()

# Think of this as a python decorator, wraps a `_fetch_X` macro with some
# logging conveniences.
#
# MACRO_NAME
#   Name of the `_fetch_X` macro.
#
#   NOTE: CMake macros execute in current scope, CMake functions create a new
#         scope. `_with_logging` checks variables it expects a `_fetch_X` macro
#         to set _in_ `_with_logging`s scope, requiring that `_fetch_X` is a
#         macro and not a function.
#
# DEP_NAME
#   Dependency name for logging
function(_with_logging MACRO_NAME DEP_NAME)
    # Set indent for logging, any logs from dep "X" will be prefixed with [x].
    set(CMAKE_MESSAGE_INDENT "[${DEP_NAME}] ")

    # CMake macros aren't textual expansions like the C preprocessor, macro vs
    # function is really just about the scope statements are executed in, so
    # calling a macro isn't a problem.
    cmake_language(CALL ${MACRO_NAME} ${ARGN})

    # reset indent.
    set(CMAKE_MESSAGE_INDENT "")

    # FetchContent_MakeAvailable(DEP) creates a dep_POPULATED variable
    # indicating an external project has been vendored in the build tree.
    #
    # WARNING: FetchContent_Declare(<name>)/FetchContent_MakeAvailable(<name>)
    #          can use anything for the name argument, if the _fetch_X macro
    #          doesn't use ${DEP_NAME} the <name>_POPULATED we're checking for
    #          here won't exist and the log may be misleading.
    string(TOLOWER ${DEP_NAME} DEP_NAME_LOWER)
    if (${DEP_NAME_LOWER}_POPULATED)
        message(STATUS "${DEP_NAME} dependency populated via FetchContent")
        message(STATUS "  Source: ${${DEP_NAME_LOWER}_SOURCE_DIR}")
        message(STATUS "  Build:  ${${DEP_NAME_LOWER}_BINARY_DIR}")
    else()
        message(STATUS "${DEP_NAME} dependency found on system via find_package")
        message(STATUS "  Config: ${${DEP_NAME}_DIR}")
    endif()
endfunction()

# GTest
#
# GTEST_VERSION
#   Version tag of GTest
macro(_fetch_GTest)
    cmake_parse_arguments(
        ARG              # prefix for parsed variables
        ""               # options (flags)
        "GTEST_VERSION"  # single-value arguments
        ""               # multi-value arguments
        ${ARGN}
    )
    if(NOT DEFINED ARG_GTEST_VERSION)
        message(FATAL_ERROR "GTEST_VERSION is required")
    endif()

	FetchContent_Declare(
		GTest
		URL https://github.com/google/googletest/archive/refs/tags/v${ARG_GTEST_VERSION}.zip
    )
	set(INSTALL_GTEST OFF)
	set(BUILD_GMOCK OFF)
	FetchContent_MakeAvailable(GTest)

    message(STATUS "TACO inner scope: gtest_POPULATED=${gtest_POPULATED}")
endmacro()

# hipdnn_frontend
#
# HIP_DNN_HASH
#   Git commit hash or tag to fetch
macro(_fetch_hipdnn_frontend)
    cmake_parse_arguments(
        ARG             # prefix for parsed variables
        ""              # options (flags)
        "HIP_DNN_HASH"  # single-value arguments
        ""              # multi-value arguments
        ${ARGN}
    )
    if(NOT DEFINED ARG_HIP_DNN_HASH)
        message(FATAL_ERROR "HIP_DNN_HASH is required")
    endif()

    FetchContent_Declare(
        hipdnn_frontend
        GIT_REPOSITORY https://github.com/ROCm/hipDNN.git
        GIT_TAG        ${ARG_HIP_DNN_HASH}
        # When FIND_PACKAGE_ARGS is passed, FetchContent_Declare tries to
        # find_package an installed version before downloading.
        FIND_PACKAGE_ARGS CONFIG
    )

    set(HIP_DNN_BUILD_BACKEND ON)
    set(HIP_DNN_BUILD_FRONTEND ON)
    set(HIP_DNN_SKIP_TESTS ON)
    set(HIP_DNN_BUILD_PLUGINS OFF)
    set(ENABLE_CLANG_TIDY OFF)
    # PIC required to link static library into shared object.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_MakeAvailable(hipdnn_frontend)
endmacro()

# IREERuntime
#
# NOTE: For now, we're not providing a FetchContent fallback for IREERuntime. It's
#       expected that the system provides this dependency. If you're running in
#       the fusilli docker container (described in sharkfuser README) passing
#       -DIREERuntime_DIR=/opt/iree/build/lib/cmake/IREE should be enough.
macro(_fetch_IREERuntime)
    find_package(IREERuntime CONFIG REQUIRED)
endmacro()

# Fusilli
#
# USE_LOCAL
#   If set, uses local source from ../sharkfuser directory. Without USE_LOCAL,
#   requires system installation via find_package.
macro(_fetch_Fusilli)
    cmake_parse_arguments(
        ARG          # prefix for parsed variables
        ""           # options (flags)
        "USE_LOCAL"  # single-value arguments
        ""           # multi-value arguments
        ${ARGN}
    )

    if(ARG_USE_LOCAL)
        message(STATUS "Using local Fusilli build from ../sharkfuser")
        FetchContent_Declare(
            Fusilli
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sharkfuser
        )
        set(FUSILLI_BUILD_TESTS OFF)
        FetchContent_MakeAvailable(Fusilli)
    else()
        find_package(Fusilli CONFIG REQUIRED)
    endif()
endmacro()
