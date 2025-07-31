# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Creates a sharkfuser C++ test.
#
#  add_sharkfuser_test(
#    NAME <test-name>
#    SRCS <file> [<file> ...]
#    [DEPS <dep> [<dep> ...]]
#  )
#
# NAME
#  The name of the executable target to create (required)
#
# SRCS
#  Source files to compile into the executable (required)
#
# DEPS
#  Additional library dependencies beyond the standard ones
#  (libfusilli and Catch2::Catch2WithMain are always linked)
function(add_sharkfuser_test)
  if(NOT SHARKFUSER_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE             # prefix
    ""                # options
    "NAME"            # one value keywords
    "SRCS;DEPS"       # multi-value keywords
    ${ARGN}           # extra arguments
  )

  _add_sharkfuser_ctest_target(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR tests
  )
endfunction()


# Creates a sharkfuser C++ sample.
#
#  add_sharkfuser_sample(
#    NAME <test-name>
#    SRCS <file> [<file> ...]
#    [DEPS <dep> [<dep> ...]]
#  )
#
# NAME
#  The name of the executable target to create (required)
#
# SRCS
#  Source files to compile into the executable (required)
#
# DEPS
#  Additional library dependencies beyond the standard ones
#  (libfusilli and Catch2::Catch2WithMain are always linked)
function(add_sharkfuser_sample)
  if(NOT SHARKFUSER_BUILD_SAMPLES)
    return()
  endif()

  cmake_parse_arguments(
    _RULE             # prefix
    ""                # options
    "NAME"            # one value keywords
    "SRCS;DEPS"       # multi-value keywords
    ${ARGN}           # extra arguments
  )

  _add_sharkfuser_ctest_target(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR samples
  )
endfunction()


# Creates a sharkfuser lit test.
#
#  add_sharkfuser_lit_test(
#    SRC <file>
#    [DEPS <dep> [<dep> ...]]
#    [TOOLS <tool> [<tool> ...]]
#  )
#
# SRC
#  The source file to compile and test (required)
#
# DEPS
#  Additional library dependencies beyond the standard ones
#  (libfusilli and Catch2::Catch2WithMain are always linked)
#
# TOOLS
#  External tools needed for the test
function(add_sharkfuser_lit_test)
  if(NOT SHARKFUSER_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE               # prefix
    ""                  # options
    "SRC"               # one value keywords
    "DEPS;TOOLS"        # multi-value keywords
    ${ARGN}             # extra arguments
  )

  if(NOT _RULE_SRC)
    message(FATAL_ERROR "add_sharkfuser_lit_test: SRC parameter is required")
  endif()

  get_filename_component(_TEST_NAME ${_RULE_SRC} NAME_WE)
  get_filename_component(_SRC_FILE_PATH ${_RULE_SRC} ABSOLUTE)

  # The executable whose output is being lit tested.
  _add_sharkfuser_executable_for_test(
    NAME ${_TEST_NAME}
    SRCS ${_RULE_SRC}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR lit
  )

  # Pass locations of tools in build directory to lit through `--path` arguments.
  set(_LIT_PATH_ARGS)
  foreach(_TOOL IN LISTS _RULE_TOOLS)
    list(APPEND _LIT_PATH_ARGS "--path" "$<TARGET_FILE_DIR:${_TOOL}>")
  endforeach()

  add_test(
    NAME ${_TEST_NAME}
    COMMAND
      ${SHARKFUSER_EXTERNAL_LIT}
      ${_LIT_PATH_ARGS}
      "--param" "TEST_EXE=$<TARGET_FILE:${_TEST_NAME}>"
      "--verbose"
      ${_SRC_FILE_PATH}
  )
endfunction()


# Creates a CTest test that wraps an executable.
#
# NAME
#  The name of the test target to create (required)
#
# SRCS
#  Source files to compile into the executable (required)
#
# DEPS
#  Additional library dependencies beyond the standard ones
#  (libfusilli and Catch2::Catch2WithMain are always linked)
#
# BIN_SUBDIR
#  Subdirectory under build/bin/ where the executable will be placed
function(_add_sharkfuser_ctest_target)
  cmake_parse_arguments(
    _RULE               # prefix
    ""                  # options
    "NAME;BIN_SUBDIR"   # one value keywords
    "SRCS;DEPS"         # multi-value keywords
    ${ARGN}             # extra arguments
  )

  # Create the target first
  _add_sharkfuser_executable_for_test(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR ${_RULE_BIN_SUBDIR}
  )

  # Add the CTest test
  add_test(NAME ${_RULE_NAME} COMMAND ${_RULE_NAME})

  # Set logging environment variables
  if(SHARKFUSER_DEBUG_BUILD)
    set_tests_properties(
      ${_RULE_NAME} PROPERTIES
      ENVIRONMENT "FUSILLI_LOG_INFO=1;FUSILLI_LOG_FILE=stdout"
    )
  endif()
endfunction()


# Creates an executable target for use in a test.
#
# NAME
#  The name of the executable target to create (required)
#
# SRCS
#  Source files to compile into the executable (required)
#
# DEPS
#  Additional library dependencies beyond the standard ones
#  (libfusilli and Catch2::Catch2WithMain are always linked)
#
# BIN_SUBDIR
#  Subdirectory under build/bin/ where the executable will be placed
function(_add_sharkfuser_executable_for_test)
  cmake_parse_arguments(
    _RULE               # prefix
    ""                  # options
    "NAME;BIN_SUBDIR"   # one value keywords
    "SRCS;DEPS"         # multi-value keywords
    ${ARGN}             # extra arguments
  )

  # Add the executable target
  add_executable(${_RULE_NAME} ${_RULE_SRCS})

  # Link libraries/dependencies
  target_link_libraries(${_RULE_NAME} PRIVATE
    ${_RULE_DEPS}
    libfusilli
    Catch2::Catch2WithMain
  )

  # Set compiler options for code coverage
  if(SHARKFUSER_CODE_COVERAGE)
    target_compile_options(${_RULE_NAME} PRIVATE -coverage -O0 -g)
    target_link_options(${_RULE_NAME} PRIVATE -coverage)
  endif()

  # Place executable in the build/bin sub-directory
  set_target_properties(
      ${_RULE_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_RULE_BIN_SUBDIR}
  )
endfunction()
