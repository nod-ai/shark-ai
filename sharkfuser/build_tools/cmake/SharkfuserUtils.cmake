# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Usage: sharkfuser_find_external_tool(TOOL_NAME [EXTRA_ERROR_MESSAGE])
macro(sharkfuser_find_external_tool TOOL_NAME)
  # Parse optional extra error message
  set(_EXTRA_ERROR_MSG "")
  if(${ARGC} GREATER 1)
    set(_EXTRA_ERROR_MSG "${ARGV1}")
  endif()

  # Convert tool name to uppercase with underscores for variable name
  string(REPLACE "-" "_" _TOOL_VAR_NAME "${TOOL_NAME}")
  string(TOUPPER "${_TOOL_VAR_NAME}" _TOOL_VAR_NAME)
  set(_FULL_VAR_NAME "SHARKFUSER_EXTERNAL_${_TOOL_VAR_NAME}")

  # Find the tool if not already set
  if(NOT ${_FULL_VAR_NAME})
    find_program(${_FULL_VAR_NAME} NAMES ${TOOL_NAME})
    if(NOT ${_FULL_VAR_NAME})
      message(FATAL_ERROR "Could not find '${TOOL_NAME}' in PATH. ${_EXTRA_ERROR_MSG}")
    endif()
  endif()
  message(STATUS "Using ${TOOL_NAME}: ${${_FULL_VAR_NAME}}")
endmacro()
