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

# Add test executable
add_executable(sharkfuser_tests
  test_tensor.cpp
  test_logging.cpp
)
target_link_libraries(sharkfuser_tests PRIVATE sharkfuser Catch2::Catch2WithMain)
add_test(NAME sharkfuser_tests COMMAND sharkfuser_tests)
set_tests_properties(
  sharkfuser_tests PROPERTIES
  ENVIRONMENT "FUSILI_LOG_INFO=1;FUSILI_LOG_FILE=stdout"
)

# Place executable in the bin directory
set_target_properties(
    sharkfuser_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
