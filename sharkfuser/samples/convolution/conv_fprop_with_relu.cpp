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

TEST_CASE("Convolution fprop; X (NHWC), W (KRSC); 1x1 conv; no padding; relu",
          "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto build_new_graph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_fprop_sample_nhwc_krsc_1x1_nopad_relu");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto X = graph->tensor(TensorAttr()
                               .setName("image")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto W = graph->tensor(TensorAttr()
                               .setName("filter")
                               .setDim({k, c, r, s})
                               .setStride({c * r * s, 1, c * s, c})); // KRSC

    auto convAttr = ConvFPropAttr()
                        .setStride({1, 1})
                        .setPadding({0, 0})
                        .setDilation({1, 1})
                        .setName("conv_fprop");

    auto convResult = graph->convFProp(X, W, convAttr);
    convResult->setName("conv_result").setDataType(DataType::Half);

    auto reluAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::RELU_FWD);
    auto reluResult = graph->pointwise(convResult, reluAttr);
    reluResult->setName("result").setOutput(true);

    // Validate, infer missing properties
    REQUIRE(isOk(graph->validate()));

    // Compile
    REQUIRE(isOk(graph->compile(handle, /*remove=*/true)));

    return std::make_tuple(graph, X, W, reluResult);
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
  auto xBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(X->getPhysicalDim()),
      /*data=*/std::vector<half>(X->getVolume(), half(1.0f)))));

  // Allocate weight buffer.
  auto wBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(W->getPhysicalDim()),
      /*data=*/std::vector<half>(W->getVolume(), half(1.0f)))));

  // Allocate output buffer.
  auto yBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(Y->getPhysicalDim()),
      /*data=*/std::vector<half>(Y->getVolume(), half(0.0f)))));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  // Execute graph once.
  REQUIRE(isOk(graph->execute(variantPack)));

  // Read output buffers.
  std::vector<half> result;
  REQUIRE(isOk(yBuf->read(handle, result)));
  for (auto val : result)
    REQUIRE(val == half(128.0f));

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    REQUIRE(isOk(graph->execute(variantPack)));

  // Repeat output buffer checks.
  result.clear();
  REQUIRE(isOk(yBuf->read(handle, result)));
  for (auto val : result)
    REQUIRE(val == half(128.0f));
}
