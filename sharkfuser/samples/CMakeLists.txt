# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# Download Catch2 for testing
if(NOT catch2_FOUND)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY  https://github.com/catchorg/Catch2.git
    GIT_TAG         ${SHARKFUSER_CATCH2_GIT_TAG}
  )
  FetchContent_MakeAvailable(catch2)
endif()

# Add sample executable
add_executable(sharkfuser_samples
  convolution/conv_fprop.cpp
)
target_link_libraries(sharkfuser_samples PRIVATE sharkfuser Catch2::Catch2WithMain)
if(SHARKFUSER_CODE_COVERAGE)
  target_compile_options(sharkfuser_samples PRIVATE -coverage -O0 -g)
  target_link_options(sharkfuser_samples PRIVATE -coverage)
endif()
add_test(NAME sharkfuser_samples COMMAND sharkfuser_samples)
if(SHARKFUSER_DEBUG_BUILD)
  set_tests_properties(
    sharkfuser_samples PROPERTIES
    ENVIRONMENT "FUSILI_LOG_INFO=1;FUSILI_LOG_FILE=stdout"
  )
endif()

# Place executable in the bin directory
set_target_properties(
    sharkfuser_samples PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
