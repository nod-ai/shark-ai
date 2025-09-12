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

TEST_CASE("Buffer allocation, move semantics and lifetime", "[buffer]") {
  // Parameterize by backend and create device-specific handles
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("gfx942 backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::GFX942)));
  }
#endif
  Handle &handle = *handlePtr;

  // Allocate a buffer of shape [2, 3] with all elements set to 1.0f (float)
  std::vector<float> data(6, 1.0f);
  Buffer buf = FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT({2, 3}), data));
  REQUIRE(buf != nullptr);

  // Read buffer and check contents
  std::vector<float> result;
  REQUIRE(isOk(buf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == 1.0f);

  // Test move semantics
  Buffer movedBuf = std::move(buf);

  // Moved-to buffer is not NULL
  REQUIRE(movedBuf != nullptr);

  // Moved-from buffer is NULL
  REQUIRE(buf == nullptr);

  // Read moved buffer and check contents
  result.clear();
  REQUIRE(isOk(movedBuf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == 1.0f);
}

TEST_CASE("Buffer import and lifetimes", "[buffer]") {
  // Parameterize by backend and create device-specific handles
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("gfx942 backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::GFX942)));
  }
#endif
  Handle &handle = *handlePtr;

  // Allocate a buffer of shape [2, 3] with all elements set to 1.0f (float)
  std::vector<float> data(6, 1.0f);
  Buffer buf = FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT({2, 3}), data));
  REQUIRE(buf != nullptr);

  // Read buffer and check contents
  std::vector<float> result;
  REQUIRE(isOk(buf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == 1.0f);

  // Test import in local scope
  {
    Buffer importedBuf = FUSILLI_REQUIRE_UNWRAP(Buffer::import(buf));
    // Both buffers co-exist and retain ownership (reference count tracked)
    REQUIRE(importedBuf != nullptr);
    REQUIRE(buf != nullptr);

    // Read imported buffer and check contents
    result.clear();
    REQUIRE(isOk(importedBuf.read(handle, result)));
    for (auto val : result)
      REQUIRE(val == 1.0f);
  }

  // Initial buffer still exists in outer scope
  REQUIRE(buf != nullptr);

  // Read original buffer and check contents
  result.clear();
  REQUIRE(isOk(buf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == 1.0f);
}

TEST_CASE("Buffer deallocation", "[buffer]") {
  // Create a handle for CPU backend
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

  std::vector<float> data(4, 2.0f);
  {
    Buffer buf = FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle, castToSizeT({2, 2}), data));
    REQUIRE(buf != nullptr);
    // Buffer will be destroyed at end of scope
  }
}
