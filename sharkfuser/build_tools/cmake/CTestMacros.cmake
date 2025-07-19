# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


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
#  (libfusili and Catch2::Catch2WithMain are always linked)
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

  add_executable(${_RULE_NAME} ${_RULE_SRCS})

  # Link libraries/dependencies
  target_link_libraries(${_RULE_NAME} PRIVATE
    ${_RULE_DEPS}
    libfusili
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


# Creates an executable target + test.
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

  # Add the test
  add_test(NAME ${_RULE_NAME} COMMAND ${_RULE_NAME})

  # Set logging environment variables
  if(SHARKFUSER_DEBUG_BUILD)
    set_tests_properties(
      ${_RULE_NAME} PROPERTIES
      ENVIRONMENT "FUSILI_LOG_INFO=1;FUSILI_LOG_FILE=stdout"
    )
  endif()
endfunction()


# Creates a sharkfuser test.
#
# NAME
#  The name of the executable target to create (required)
#
# SRCS
#  Source files to compile into the executable (required)
#
# DEPS
#  Additional library dependencies beyond the standard ones
#  (libfusili and Catch2::Catch2WithMain are always linked)
function(add_sharkfuser_test)
  if(NOT SHARKFUSER_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  _add_sharkfuser_ctest_target(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR tests
  )
endfunction()


# Creates a sharkfuser sample.
#
# NAME
#  The name of the executable target to create (required)
#
# SRCS
#  Source files to compile into the executable (required)
#
# DEPS
#  Additional library dependencies beyond the standard ones
#  (libfusili and Catch2::Catch2WithMain are always linked)
function(add_sharkfuser_sample)
  if(NOT SHARKFUSER_BUILD_SAMPLES)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  _add_sharkfuser_ctest_target(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR samples
  )
endfunction()


# Creates a lit test that compiles a source file and runs lit on it.
#
#  add_sharkfuser_lit_test(
#    SRC <source-file>
#    [TOOLS <tool1> <tool2> ...]
#    [DEPS <dep1> <dep2> ...]
#  )
#
# SRC
#  The source file to compile and test (required)
#
# TOOLS
#  External tools needed for the test (e.g., FileCheck)
#
# DEPS
#  Library dependencies for the compiled executable
function(add_sharkfuser_lit_test)
  if(NOT SHARKFUSER_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE               # prefix
    ""                  # options
    "SRC"               # one value keywords
    "TOOLS;DEPS"        # multi-value keywords
    ${ARGN}
  )
  if(NOT _RULE_SRC)
    message(FATAL_ERROR "add_sharkfuser_lit_test: SRC parameter is required")
  endif()

  get_filename_component(_ABSOLUTE_RULE_SRC ${_RULE_SRC} ABSOLUTE)
  get_filename_component(_TEST_NAME ${_RULE_SRC} NAME_WE)

  # The executable who's output is being lit tested.
  _add_sharkfuser_executable_for_test(
    NAME ${_TEST_NAME}
    SRCS ${_RULE_SRC}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR lit
  )

  # Pass lit locations of tools in build directory through `--path` arguments.
  set(_LIT_PATH_ARGS
    "--path" "$<TARGET_FILE_DIR:${_TEST_NAME}>" # include test itself
    "--path" "$<TARGET_FILE_DIR:FileCheck>"     # include FileCheck by default
    )
  foreach(_TOOL IN LISTS _RULE_TOOLS)
    list(APPEND _LIT_PATH_ARGS "--path" "$<TARGET_FILE_DIR:${_TOOL}>")
  endforeach()

  add_test(
    NAME
      ${_TEST_NAME}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${LLVM_EXTERNAL_LIT}"
      "${_ABSOLUTE_RULE_SRC}"
      ${_LIT_PATH_ARGS}
      # lit config sets the "%test_exe" substitution based on this param.
      "--param" "TEST_EXE=$<TARGET_FILE:${_TEST_NAME}>"
      # Ensures lit prints a (more) useful error message on failure.
      "--verbose"
  )

  # Apparently this flag helps FileCheck spit out nicer error messages.
  set_tests_properties(${_TEST_NAME} PROPERTIES ENVIRONMENT
    "FILECHECK_OPTS=--enable-var-scope")

  # Dependencies for the test.
  if(_RULE_TOOLS)
    set_tests_properties(${_TEST_NAME} PROPERTIES DEPENDS "${_RULE_TOOLS}")
  endif()
endfunction()
