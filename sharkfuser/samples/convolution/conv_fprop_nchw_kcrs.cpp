// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Convolution fprop; X (NCHW), W (KCRS); 1x1 conv; no padding",
          "[conv][graph]") {
  int64_t n = 4, c = 4, h = 4, w = 4, k = 4, r = 1, s = 1;

  auto build_new_graph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_fprop_sample_nchw_kcrs_1x1_nopad");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto X = graph->tensor(TensorAttr()
                               .setName("image")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, h * w, w, 1})); // NCHW

    auto W = graph->tensor(TensorAttr()
                               .setName("filter")
                               .setDim({k, c, r, s})
                               .setStride({c * r * s, r * s, s, 1})); // KCRS

    auto convAttr = ConvFPropAttr()
                        .setPadding({0, 0})
                        .setStride({1, 1})
                        .setDilation({1, 1})
                        .setName("conv_fprop");

    auto Y = graph->convFProp(X, W, convAttr);
    Y->setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, X, W, Y);
  };

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

  // Build graph for the given handle (device), validate and compile it.
  auto [graph, X, W, Y] = build_new_graph(handle);

  // Allocate input buffer.
  auto xBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, X, DataType::Half, 1.0f));

  // Allocate weight buffer.
  auto wBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, W, DataType::Half, 1.0f));

  // Allocate output buffer.
  auto yBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, Y, DataType::Half, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(variantPack));

  // Read output buffers.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(4.0f));

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(variantPack));

  // Repeat output buffer checks.
  result.clear();
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(4.0f));
}
