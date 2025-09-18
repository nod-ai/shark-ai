/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/LoggingUtils.hpp>

#define IREE_PLUGIN_TESTS "iree_integration_test"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  hipdnn_sdk::test_utilities::initializeSpdlogDefaultLogger(IREE_PLUGIN_TESTS);

  return RUN_ALL_TESTS();
}
