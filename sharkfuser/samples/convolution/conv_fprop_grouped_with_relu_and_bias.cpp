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

TEST_CASE("Convolution fprop; grouped; X (NHWC), W (KRSC); 1x1 conv; no "
          "padding; relu; bias",
          "[conv][graph]") {
  constexpr int64_t n = 16, c = 128, h = 64, w = 64, k = 256, fc = 16, r = 1,
                    s = 1;

  auto build_new_graph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_fprop_grouped_sample_nhwc_krsc_1x1_nopad_bias_relu");
    graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

    auto X = graph->tensor(TensorAttr()
                               .setName("image")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto W = graph->tensor(TensorAttr()
                               .setName("filter")
                               .setDim({k, fc, r, s})
                               .setStride({fc * r * s, 1, fc * s, fc})); // KRSC

    auto convAttr = ConvFPropAttr()
                        .setStride({1, 1})
                        .setPadding({0, 0})
                        .setDilation({1, 1})
                        .setName("conv_fprop");

    auto convResult = graph->convFProp(X, W, convAttr);
    convResult->setName("conv_result").setDataType(DataType::Half);

    auto B = graph->tensor(TensorAttr()
                               .setName("bias")
                               .setDim({1, k, 1, 1})
                               .setStride({k, 1, k, k}));
    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    auto biasResult = graph->pointwise(convResult, B, biasAttr);
    biasResult->setName("bias_result").setDataType(DataType::Half);

    auto reluAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::RELU_FWD);
    auto reluResult = graph->pointwise(biasResult, reluAttr);
    reluResult->setName("result").setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, X, W, B, reluResult);
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
  auto [graph, X, W, B, Y] = build_new_graph(handle);

  // Allocate input, weights and bias buffers.
  constexpr float inputScalar = 1.0f;
  auto xBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, X, DataType::Half, inputScalar));
  auto wBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, W, DataType::Half, inputScalar));
  auto bBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, B, DataType::Half, inputScalar));
  // Allocate output buffer.
  auto yBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, Y, DataType::Half, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {B, bBuf},
          {Y, yBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Calculate expected output value.
  constexpr half expected = std::max(
      static_cast<float>(fc * r * s) * inputScalar * inputScalar + inputScalar,
      0.0f);

  // Read output buffers.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  result.clear();
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == expected);
}
