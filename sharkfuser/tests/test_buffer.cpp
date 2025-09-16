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
  // Parameterize by backend and create device-specific handles.
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

  // Allocate a buffer of shape [2, 3] with all elements set to 1.0f (float).
  std::vector<float> data(6, 1.0f);
  Buffer buf = FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT({2, 3}), data));
  REQUIRE(buf != nullptr);

  // Read buffer and check contents.
  std::vector<float> result;
  REQUIRE(isOk(buf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == 1.0f);

  // Test move semantics.
  Buffer movedBuf = std::move(buf);

  // Moved-to buffer is not NULL.
  // Moved-from buffer is NULL.
  REQUIRE(movedBuf != nullptr);
  REQUIRE(buf == nullptr);

  // Read moved buffer and check contents.
  result.clear();
  REQUIRE(isOk(movedBuf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == 1.0f);
}

TEST_CASE("Buffer import and lifetimes", "[buffer]") {
  // Parameterize by backend and create device-specific handles.
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

  // Allocate a buffer of shape [2, 3] with all elements set to half(1.0f).
  std::vector<half> data(6, half(1.0f));
  Buffer buf = FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT({2, 3}), data));
  REQUIRE(buf != nullptr);

  // Read buffer and check contents.
  std::vector<half> result;
  REQUIRE(isOk(buf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == half(1.0f));

  // Test import in local scope.
  {
    Buffer importedBuf = FUSILLI_REQUIRE_UNWRAP(Buffer::import(buf));
    // Both buffers co-exist and retain ownership (reference count tracked).
    REQUIRE(importedBuf != nullptr);
    REQUIRE(buf != nullptr);

    // Read imported buffer and check contents.
    result.clear();
    REQUIRE(isOk(importedBuf.read(handle, result)));
    for (auto val : result)
      REQUIRE(val == half(1.0f));
  }

  // Initial buffer still exists in outer scope.
  REQUIRE(buf != nullptr);

  // Read original buffer and check contents.
  result.clear();
  REQUIRE(isOk(buf.read(handle, result)));
  for (auto val : result)
    REQUIRE(val == 1.0f);
}

TEST_CASE("Buffer errors", "[buffer]") {
  SECTION("Import NULL buffer") {
    // Importing a NULL buffer view should fail.
    iree_hal_buffer_view_t *nullBuf = nullptr;
    ErrorObject status = Buffer::import(nullBuf);
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status.getMessage() ==
            "Buffer::import failed as externalBufferView* is NULL");
  }

  SECTION("Reading into a non-empty vector") {
    // Reading into a non-empty vector should fail.
    Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

    // Allocate a buffer of shape [2, 3] with all elements set to 1.0f (float).
    std::vector<float> data(6, 0.0f);
    Buffer buf = FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle, castToSizeT({2, 3}), data));

    // Read buffer into a non-empty vector.
    std::vector<float> result(6, 1.0f);
    REQUIRE(!result.empty());
    ErrorObject status = buf.read(handle, result);
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status.getMessage() ==
            "Buffer::read failed as outData is NOT empty");

    // Read buffer into an empty vector should work.
    result.clear();
    REQUIRE(isOk(buf.read(handle, result)));
    for (auto val : result)
      REQUIRE(val == 0.0f);
  }
}
