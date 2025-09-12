// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("Buffer allocation and lifetime", "[buffer]") {
  // Create a handle for CPU backend
  auto handleOr = Handle::create(Backend::CPU);
  REQUIRE(isOk(handleOr));
  Handle &handle = *handleOr;

  // Allocate a buffer of shape [2, 3] with all elements set to 1.0f (float)
  std::vector<float> data(6, 1.0f);
  auto bufOr = Buffer::allocate(handle, castToSizeT({2, 3}), data);
  REQUIRE(isOk(bufOr));
  Buffer &buf = *bufOr;

  // The buffer view pointer should not be null
  REQUIRE(buf != nullptr);

  // Test move semantics
  Buffer movedBuf = std::move(buf);
  REQUIRE(movedBuf != nullptr);

  // After move, original buffer should be in a valid but unspecified state
  // (Optional: check that moved-from buffer does not crash)
}

TEST_CASE("Buffer deallocation", "[buffer]") {
  auto handleOr = Handle::create(Backend::CPU);
  REQUIRE(isOk(handleOr));
  Handle &handle = *handleOr;

  std::vector<float> data(4, 2.0f);
  {
    auto bufOr = Buffer::allocate(handle, castToSizeT({2, 2}), data);
    REQUIRE(isOk(bufOr));
    Buffer &buf = *bufOr;
    REQUIRE(buf != nullptr);
    // Buffer will be destroyed at end of scope
  }
}
