# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(GoogleTest)
find_package(Threads REQUIRED)

add_executable(iree_plugin_integration_test
    main.cpp
    ConvFpropIntegrationTest.cpp
)

target_compile_options(iree_plugin_integration_test PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
target_compile_definitions(iree_plugin_integration_test PRIVATE
    PLUGIN_DIR="${IREE_PLUGIN_DIR}/"
)

target_link_libraries(iree_plugin_integration_test PUBLIC
    GTest::gtest
    hip::host
    hipdnn_frontend
    hipdnn_sdk
    Threads::Threads
)

# Register test with CTest
gtest_discover_tests(iree_plugin_integration_test)

if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(iree_plugin_integration_test)

    add_integration_test_target(iree_plugin_integration_test ${CMAKE_CURRENT_BINARY_DIR})
endif()
