# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


function(_add_sharkfuser_target)
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

  add_test(NAME ${_RULE_NAME} COMMAND ${_RULE_NAME})

  # Set logging environment variables
  if(SHARKFUSER_DEBUG_BUILD)
    set_tests_properties(
      ${_RULE_NAME} PROPERTIES
      ENVIRONMENT "FUSILI_LOG_INFO=1;FUSILI_LOG_FILE=stdout"
    )
  endif()

  # Place executable in the build/bin sub-directory
  set_target_properties(
      ${_RULE_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_RULE_BIN_SUBDIR}
  )
endfunction()


function(add_sharkfuser_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  if(NOT SHARKFUSER_BUILD_TESTS)
    return()
  endif()

  _add_sharkfuser_target(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR tests
  )
endfunction()


function(add_sharkfuser_sample)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  if(NOT SHARKFUSER_BUILD_SAMPLES)
    return()
  endif()

  _add_sharkfuser_target(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR samples
  )
endfunction()
