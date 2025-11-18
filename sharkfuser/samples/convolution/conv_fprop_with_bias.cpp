// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Convolution fprop; X (NHWC), W (KRSC); 1x1 conv; no padding; bias",
          "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_fprop_sample_nhwc_krsc_1x1_nopad_bias");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto xT = graph->tensor(TensorAttr()
                                .setName("image")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto wT = graph->tensor(TensorAttr()
                                .setName("filter")
                                .setDim({k, c, r, s})
                                .setStride({c * r * s, 1, c * s, c})); // KRSC

    auto convAttr = ConvFPropAttr()
                        .setStride({1, 1})
                        .setPadding({0, 0})
                        .setDilation({1, 1})
                        .setName("conv_fprop");

    auto convResult = graph->convFProp(xT, wT, convAttr);
    convResult->setName("conv_result").setDataType(DataType::Half);

    auto bT = graph->tensor(TensorAttr()
                                .setName("bias")
                                .setDim({1, k, 1, 1})
                                .setStride({k, 1, k, k}));
    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    auto biasResult = graph->pointwise(convResult, bT, biasAttr);
    biasResult->setName("result").setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, xT, wT, bT, biasResult);
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
  auto [graph, xT, wT, bT, yT] = buildNewGraph(handle);

  // Allocate input buffer.
  auto xBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, xT, DataType::Half, 1.0f));
  // Allocate weight buffer.
  auto wBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, wT, DataType::Half, 1.0f));
  // Allocate bias buffer.
  auto bBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, bT, DataType::Half, 1.0f));
  // Allocate output buffer.
  auto yBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, yT, DataType::Half, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {wT, wBuf},
          {bT, bBuf},
          {yT, yBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffers.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(129.0f));

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  result.clear();
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(129.0f));
}
