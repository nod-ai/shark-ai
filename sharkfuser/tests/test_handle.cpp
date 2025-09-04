// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

using namespace fusilli;

TEST_CASE("Single FusilliHandle creation", "[handle]") {
  SECTION("CPU handle") {
    FusilliHandle handle =
        FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("GPU handle") {
    FusilliHandle handle =
        FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::GFX942));
  }
#endif
}

TEST_CASE("Multiple FusilliHandle creation", "[handle]") {
  FusilliHandle handle1 =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::CPU));
#ifdef FUSILLI_ENABLE_AMDGPU
  FusilliHandle handle2 =
      FUSILLI_REQUIRE_UNWRAP(FusilliHandle::create(Backend::GFX942));
#endif
}

TEST_CASE("Multi-threaded FusilliHandle creation", "[handle][thread]") {
  constexpr int kNumThreads = 32;

  std::vector<FusilliHandle> handles;
  handles.reserve(kNumThreads);

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::mutex handlesMutex;
  std::atomic<bool> creationFailed{false};
  std::atomic<bool> startFlag{false};

  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      // Wait for start signal
      while (!startFlag.load()) {
      }
      auto handleOrError = FusilliHandle::create(Backend::CPU);
      if (isError(handleOrError)) {
        creationFailed.store(true);
        return;
      }
      std::lock_guard<std::mutex> lock(handlesMutex);
      handles.push_back(std::move(*handleOrError));
    });
  }
  // Start all threads simultaneously
  startFlag.store(true);
  for (auto &t : threads)
    t.join();

  REQUIRE(!creationFailed.load());
  REQUIRE(handles.size() == kNumThreads);
}
