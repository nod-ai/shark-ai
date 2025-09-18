# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
#     find_package(GTest CONFIG REQUIRED)
# endif()

include(GoogleTest)
find_package(Threads REQUIRED)

if(HIP_DNN_SKIP_TESTS)
    message(STATUS "Skipping relu toy plugin integration tests.")
    return()
endif()

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


if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(iree_plugin_integration_test)

    add_integration_test_target(iree_plugin_integration_test ${CMAKE_CURRENT_BINARY_DIR})
endif()
