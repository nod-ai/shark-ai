// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <atomic>
#include <barrier>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace fusilli;

TEST_CASE("Single Handle creation", "[handle]") {
  SECTION("CPU handle") {
    Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("GPU handle") {
    Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU));
  }
  SECTION("GPU handle on device") {
    Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU, 0));
  }
  SECTION("GPU handle on device with stream") {
    Handle handle =
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU, 0, 0));
  }
#endif
}

TEST_CASE("Multiple Handle creation", "[handle]") {
  Handle handle1 = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle2 = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU));
#endif
}

TEST_CASE("Multi-threaded Handle creation", "[handle][thread]") {
  constexpr int kNumThreads = 32;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::vector<Handle> handles;
  handles.reserve(kNumThreads);

  // Create a barrier to force threads to start simultaneously.
  std::barrier startBarrier(kNumThreads);

  // Atomic flag to track failures during handle creation.
  std::atomic<bool> creationFailed{false};

  // Mutex for pushing to handles in a thread-safe manner.
  std::mutex handlesMutex;

  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      // Wait at the barrier until all threads reach this point.
      startBarrier.arrive_and_wait();
      // Create the handle.
      auto handleOrError = Handle::create(Backend::CPU);
      if (isError(handleOrError)) {
        creationFailed.store(true);
        return;
      }
      std::lock_guard<std::mutex> lock(handlesMutex);
      handles.push_back(std::move(*handleOrError));
    });
  }
  // Wait for all threads to finish.
  for (auto &t : threads)
    t.join();

  REQUIRE(!creationFailed.load());
  REQUIRE(handles.size() == kNumThreads);
}

TEST_CASE("Handle creation with deviceId and stream, CPU backend should fail",
          "[handle]") {
  // Attempting to create CPU handle with with a specific deviceId and stream
  // should fail.
  auto handleOrError =
      Handle::create(Backend::CPU, /*deviceId=*/0, /*stream=*/0);
  REQUIRE(isError(handleOrError));
  ErrorObject error = handleOrError;
  REQUIRE(error.getCode() == ErrorCode::InvalidArgument);
}

TEST_CASE("Handle creation with deviceId, CPU backend should fail",
          "[handle]") {
  // Attempting to create CPU handle with with a specific deviceId should fail.
  auto handleOrError = Handle::create(Backend::CPU, /*deviceId=*/0);
  REQUIRE(isError(handleOrError));
  ErrorObject error = handleOrError;
  REQUIRE(error.getCode() == ErrorCode::InvalidArgument);
}
