// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_all.hpp>
#include <memory>
#include <vector>

#include "conv_sample_utils.h"
#include "utils.h"

using namespace fusilli;
using namespace fusilli_conv_samples;

TEST_CASE("Convolution fprop; X (NHWC), W (KRSC); 1x1 conv; no padding",
          "[conv][graph]") {
  auto config = GENERATE(values<ConvSampleConfig>({
      {
          .n = 16,
          .c = 128,
          .h = 64,
          .w = 64,
          .k = 256,
          .r = 1,
          .s = 1,
          .layout = ConvSampleLayout::NHWC_KRSC,
          .expected = half(128.0f),
      },
  }));

  // Parameterize sample by backend and create device-specific handles.
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

  // Build graph an inputs.
  auto sample = buildSample(handle, config);

  // Execute graph once.
  FUSILLI_REQUIRE_OK(sample.graph.execute(sample.variantPack));

  // Read output buffers.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(sample.yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == config.expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(sample.graph.execute(sample.variantPack));

  // Repeat output buffer checks.
  result.clear();
  FUSILLI_REQUIRE_OK(sample.yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == config.expected);
}
