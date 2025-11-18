# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Creates multiple fusilli C++ tests.
#
#  add_fusilli_tests(
#    PREFIX <test-name-prefix>
#    SRCS <file> [<file> ...]
#    [DEPS <dep> [<dep> ...]]
#  )
#
# PREFIX
#  A prefix for the executable target to create (required). Each target will be
#  suffixed with the file name.
#
# SRCS
#  Source files to compile into individual executables (required).
#
# DEPS
#  Library dependencies to be linked to the targets.
function(add_fusilli_tests)
  if(NOT FUSILLI_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE             # prefix
    ""                # options
    "PREFIX"          # one value keywords
    "SRCS;DEPS"       # multi-value keywords
    ${ARGN}           # extra arguments
  )

  if(NOT DEFINED _RULE_PREFIX)
    message(FATAL_ERROR "add_fusilli_tests: PREFIX is required")
  endif()

  if(NOT DEFINED _RULE_SRCS)
    message(FATAL_ERROR "add_fusilli_tests: SRCS is required")
  endif()

  foreach(_TEST_FILE ${_RULE_SRCS})
    # Extract the base name from the file path for unique naming
    get_filename_component(_FILE_NAME ${_TEST_FILE} NAME_WE)
    set(_TEST_NAME "${_RULE_PREFIX}_${_FILE_NAME}")

    _add_fusilli_ctest_target(
      NAME ${_TEST_NAME}
      SRCS ${_TEST_FILE}
      DEPS ${_RULE_DEPS}
      BIN_SUBDIR tests
    )
  endforeach()
endfunction()

# Creates multiple fusilli C++ samples.
#
#  add_fusilli_samples(
#    PREFIX <sample-name-prefix>
#    SRCS <file> [<file> ...]
#    [DEPS <dep> [<dep> ...]]
#  )
#
# PREFIX
#  A prefix for the executable target to create (required). Each target will be
#  suffixed with the file name.
#
# SRCS
#  Source files to compile into individual executables (required).
#
# DEPS
#  Library dependencies to be linked to the targets.
function(add_fusilli_samples)
  if(NOT FUSILLI_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE             # prefix
    ""                # options
    "PREFIX"          # one value keywords
    "SRCS;DEPS"       # multi-value keywords
    ${ARGN}           # extra arguments
  )

  if(NOT DEFINED _RULE_PREFIX)
    message(FATAL_ERROR "add_fusilli_samples: PREFIX is required")
  endif()

  if(NOT DEFINED _RULE_SRCS)
    message(FATAL_ERROR "add_fusilli_samples: SRCS is required")
  endif()

  foreach(_SAMPLE_FILE ${_RULE_SRCS})
    # Extract the base name from the file path for unique naming
    get_filename_component(_FILE_NAME ${_SAMPLE_FILE} NAME_WE)
    set(_SAMPLE_NAME "${_RULE_PREFIX}_${_FILE_NAME}")

    _add_fusilli_ctest_target(
      NAME ${_SAMPLE_NAME}
      SRCS ${_SAMPLE_FILE}
      DEPS ${_RULE_DEPS}
      BIN_SUBDIR samples
    )
  endforeach()
endfunction()

# Creates a fusilli C++ benchmark.
#
#  add_fusilli_benchmark(
#    NAME <benchmark-name>
#    SRCS <file> [<file> ...]
#    [DEPS <dep> [<dep> ...]]
#    ARGS <args>
#  )
#
# NAME
#  The name of the executable target to create (required).
#
# DRIVER
#  The benchmark driver to use (required).
#
# ARGS
#  Arguments to the benchmark driver (required).
function(add_fusilli_benchmark)
  if(NOT FUSILLI_BUILD_BENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE               # prefix
    ""                  # options
    "NAME;DRIVER"       # one value keywords
    "ARGS"              # multi-value keywords
    ${ARGN}             # extra arguments
  )

  if(NOT DEFINED _RULE_NAME)
    message(FATAL_ERROR "add_fusilli_benchmark: NAME is required")
  endif()

  if(NOT DEFINED _RULE_DRIVER)
    message(FATAL_ERROR "add_fusilli_benchmark: DRIVER is required")
  endif()

  if(NOT DEFINED _RULE_ARGS)
    message(FATAL_ERROR "add_fusilli_benchmark: ARGS is required")
  endif()

  add_test(NAME ${_RULE_NAME} COMMAND ${_RULE_DRIVER} ${_RULE_ARGS})

  # Configure cache dir and logging flags.
  # Pass `FUSILLI_CACHE_DIR=/tmp` to configure the compilation cache to be
  # written to /tmp. It defaults to $HOME when not set but there are
  # permissions issues with GitHub Actions CI runners when accessing $HOME.
  set(_ENV_VARS "FUSILLI_CACHE_DIR=/tmp")
  if(FUSILLI_ENABLE_LOGGING)
    list(APPEND _ENV_VARS "FUSILLI_LOG_INFO=1" "FUSILLI_LOG_FILE=stdout")
  endif()

  # Set environment variables for test
  set_tests_properties(
    ${_RULE_NAME} PROPERTIES
    ENVIRONMENT "${_ENV_VARS}"
  )
endfunction()


# Creates a fusilli lit test.
#
#  add_fusilli_lit_test(
#    SRC <file>
#    [DEPS <dep> [<dep> ...]]
#    [TOOLS <tool> [<tool> ...]]
#  )
#
# SRC
#  The source file to compile and test (required).
#
# DEPS
#  Library dependencies to be linked to this target.
#
# TOOLS
#  External tools needed for the test.
function(add_fusilli_lit_test)
  if(NOT FUSILLI_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE               # prefix
    ""                  # options
    "SRC"               # one value keywords
    "DEPS;TOOLS"        # multi-value keywords
    ${ARGN}             # extra arguments
  )

  if(NOT DEFINED _RULE_SRC)
    message(FATAL_ERROR "add_fusilli_lit_test: SRC is required")
  endif()

  get_filename_component(_TEST_NAME ${_RULE_SRC} NAME_WE)
  get_filename_component(_SRC_FILE_PATH ${_RULE_SRC} ABSOLUTE)

  # The executable whose output is being lit tested.
  _add_fusilli_executable_for_test(
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

  # Configure CHECK prefix for backend-specific lit tests
  if(FUSILLI_SYSTEMS_AMDGPU)
    set(_BACKEND_VALUE "AMDGPU")
  else()
    set(_BACKEND_VALUE "CPU")
  endif()

  add_test(
    NAME ${_TEST_NAME}
    COMMAND
      ${FUSILLI_EXTERNAL_lit}
      ${_LIT_PATH_ARGS}
      "--param" "TEST_EXE=$<TARGET_FILE:${_TEST_NAME}>"
      "--param" "BACKEND=${_BACKEND_VALUE}"
      "--verbose"
      ${_SRC_FILE_PATH}
  )
endfunction()


# Creates a CTest test that wraps an executable.
#
# NAME
#  The name of the test target to create (required).
#
# SRCS
#  Source files to compile into the executable (required).
#
# DEPS
#  Library dependencies to be linked to this target.
#
# BIN_SUBDIR
#  Subdirectory under build/bin/ where the executable will be placed.
function(_add_fusilli_ctest_target)
  cmake_parse_arguments(
    _RULE                 # prefix
    ""                    # options
    "NAME;BIN_SUBDIR"     # one value keywords
    "SRCS;DEPS" # multi-value keywords
    ${ARGN}               # extra arguments
  )

  # Create the target first.
  _add_fusilli_executable_for_test(
    NAME ${_RULE_NAME}
    SRCS ${_RULE_SRCS}
    DEPS ${_RULE_DEPS}
    BIN_SUBDIR ${_RULE_BIN_SUBDIR}
  )

  # Add the CTest test.
  add_test(NAME ${_RULE_NAME} COMMAND ${_RULE_NAME})

  # Configure cache dir and logging flags.
  # Pass `FUSILLI_CACHE_DIR=/tmp` to configure the compilation cache to be
  # written to /tmp. It defaults to $HOME when not set but there are
  # permissions issues with GitHub Actions CI runners when accessing $HOME.
  set(_ENV_VARS "FUSILLI_CACHE_DIR=/tmp")
  if(FUSILLI_ENABLE_LOGGING)
    list(APPEND _ENV_VARS "FUSILLI_LOG_INFO=1" "FUSILLI_LOG_FILE=stdout")
  endif()

  # Set environment variables for test
  set_tests_properties(
    ${_RULE_NAME} PROPERTIES
    ENVIRONMENT "${_ENV_VARS}"
  )
endfunction()


# Creates an executable target for use in a test.
#
# NAME
#  The name of the executable target to create (required).
#
# SRCS
#  Source files to compile into the executable (required).
#
# DEPS
#  Library dependencies to be linked to this target.
#
# BIN_SUBDIR
#  Subdirectory under build/bin/ where the executable will be placed.
function(_add_fusilli_executable_for_test)
  cmake_parse_arguments(
    _RULE               # prefix
    ""                  # options
    "NAME;BIN_SUBDIR"   # one value keywords
    "SRCS;DEPS"         # multi-value keywords
    ${ARGN}             # extra arguments
  )

  # Add the executable target.
  add_executable(${_RULE_NAME} ${_RULE_SRCS})

  # Link libraries/dependencies.
  target_link_libraries(${_RULE_NAME} PRIVATE
    ${_RULE_DEPS}
  )

  # Set compiler options for code coverage.
  if(FUSILLI_CODE_COVERAGE)
    # The `-fprofile-update=atomic` flag tells GCC to use atomic updates
    # to .gcda files to avoid race conditions in concurrent environments.
    # Without this, coverage may fail with:
    #   geninfo: ERROR: Unexpected negative count '-1' for /usr/include/c++/13/bits/hashtable.h:1964
    target_compile_options(${_RULE_NAME} PRIVATE -coverage -fprofile-update=atomic -O0 -g)
    target_link_options(${_RULE_NAME} PRIVATE -coverage)
  endif()

  # Place executable in the build/bin sub-directory.
  set_target_properties(
      ${_RULE_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${_RULE_BIN_SUBDIR}
  )
endfunction()
