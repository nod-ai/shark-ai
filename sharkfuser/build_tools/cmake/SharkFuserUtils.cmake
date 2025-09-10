# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Usage: sharkfuser_find_program(TOOL_NAME [EXTRA_ERROR_MESSAGE])
macro(sharkfuser_find_program TOOL_NAME)
  # Parse optional extra error message
  set(_EXTRA_ERROR_MSG "")
  if(${ARGC} GREATER 1)
    set(_EXTRA_ERROR_MSG "${ARGV1}")
  endif()

  # Replace hyphens in tool name with underscores. Cache variables can be set
  # through the shell, where hyphens are invalid in variable names.
  string(REPLACE "-" "_" _TOOL_VAR_NAME "${TOOL_NAME}")
  set(_FULL_VAR_NAME "SHARKFUSER_EXTERNAL_${_TOOL_VAR_NAME}")

  # Find the tool if not already set
  if(NOT ${_FULL_VAR_NAME})
    find_program(${_FULL_VAR_NAME} NAMES ${TOOL_NAME})
    if(NOT ${_FULL_VAR_NAME})
      message(FATAL_ERROR "Could not find '${TOOL_NAME}' in PATH. ${_EXTRA_ERROR_MSG}")
    endif()
  endif()
  message(STATUS "Using ${TOOL_NAME}: ${${_FULL_VAR_NAME}}")
  add_executable(${TOOL_NAME} IMPORTED GLOBAL)
  set_target_properties(${TOOL_NAME} PROPERTIES IMPORTED_LOCATION "${${_FULL_VAR_NAME}}")
endmacro()
