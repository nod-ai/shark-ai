// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <iree/runtime/api.h>

#include <memory>
#include <utility>
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
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
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
  FUSILLI_REQUIRE_OK(buf.read(handle, result));
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
  FUSILLI_REQUIRE_OK(movedBuf.read(handle, result));
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
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
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
  FUSILLI_REQUIRE_OK(buf.read(handle, result));
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
    FUSILLI_REQUIRE_OK(importedBuf.read(handle, result));
    for (auto val : result)
      REQUIRE(val == half(1.0f));
  }

  // Initial buffer still exists in outer scope.
  REQUIRE(buf != nullptr);

  // Read original buffer and check contents.
  result.clear();
  FUSILLI_REQUIRE_OK(buf.read(handle, result));
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
    FUSILLI_REQUIRE_OK(buf.read(handle, result));
    for (auto val : result)
      REQUIRE(val == 0.0f);
  }

  SECTION("Buffer allocation with mismatched data size") {
    Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

    // Test case 1: bufferData has more elements than bufferShape expects.
    std::vector<float> tooMuchData(10, 1.0f);
    ErrorObject status1 =
        Buffer::allocate(handle, castToSizeT({2, 3}), tooMuchData);
    REQUIRE(isError(status1));
    REQUIRE(status1.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status1.getMessage() ==
            "Buffer::allocate failed: bufferData size (10) does not match "
            "product of bufferShape dimensions (6)");

    // Test case 2: bufferData has fewer elements than bufferShape expects.
    std::vector<float> tooLittleData(4, 1.0f);
    ErrorObject status2 =
        Buffer::allocate(handle, castToSizeT({2, 3}), tooLittleData);
    REQUIRE(isError(status2));
    REQUIRE(status2.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status2.getMessage() ==
            "Buffer::allocate failed: bufferData size (4) does not match "
            "product of bufferShape dimensions (6)");
  }

  SECTION("Buffer allocation with zero dimension") {
    Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

    // Test case 1: bufferShape with a zero dimension.
    std::vector<float> someData(5, 1.0f);
    ErrorObject status1 =
        Buffer::allocate(handle, castToSizeT({2, 0, 3}), someData);
    REQUIRE(isError(status1));
    REQUIRE(status1.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status1.getMessage() ==
            "Buffer::allocate failed: cannot allocate a buffer with zero size");

    // Test case 2: Empty bufferData and empty bufferShape.
    std::vector<float> noData;
    ErrorObject status2 = Buffer::allocate(handle, castToSizeT({}), noData);
    REQUIRE(isError(status2));
    REQUIRE(status2.getCode() == ErrorCode::RuntimeFailure);
    REQUIRE(status2.getMessage() ==
            "Buffer::allocate failed: cannot allocate a buffer with zero size");
  }
}
