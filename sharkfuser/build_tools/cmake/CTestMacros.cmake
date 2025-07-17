# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


function(add_sharkfuser_test)
  cmake_parse_arguments(
    _RULE           # prefix
    ""              # options
    "NAME"          # one value keywords
    "SRCS;DEPS"     # multi-value keywords
    ${ARGN}         # other arguments
  )

  if(NOT SHARKFUSER_BUILD_TESTS)
    return()
  endif()

  add_executable(${_RULE_NAME} ${_RULE_SRCS})

  # Link libraries/dependencies
  target_link_libraries(${_RULE_NAME} PRIVATE
    ${_RULE_DEPS}
    ${SHARKFUSER_LINK_LIBRARY_NAME}
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

  # Place executable in the bin directory
  set_target_properties(
      ${_RULE_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endfunction()


function(add_sharkfuser_sample)
  cmake_parse_arguments(
    _RULE           # prefix
    ""              # options
    "NAME"          # one value keywords
    "SRCS;DEPS"     # multi-value keywords
    ${ARGN}         # other arguments
  )

  if(NOT SHARKFUSER_BUILD_SAMPLES)
    return()
  endif()

  add_executable(${_RULE_NAME} ${_RULE_SRCS})

  # Link libraries/dependencies
  target_link_libraries(${_RULE_NAME} PRIVATE
    ${_RULE_DEPS}
    ${SHARKFUSER_LINK_LIBRARY_NAME}
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

  # Place executable in the bin directory
  set_target_properties(
      ${_RULE_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endfunction()
