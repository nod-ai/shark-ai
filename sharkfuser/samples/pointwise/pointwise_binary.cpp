// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Pointwise Binary modes", "[pointwise][graph]") {
  int64_t n = 2, c = 16, h = 64, w = 64;

  auto execute = [=]<typename T>(const std::shared_ptr<Handle> &handlePtr,
                                 PointwiseAttr::Mode mode, DataType dt, T x0,
                                 T x1) {
    auto build_new_graph = [=](const Handle &handle) {
      auto graph = std::make_shared<Graph>();
      graph->setName("pointwise_binary");
      graph->setIODataType(dt).setComputeDataType(dt);

      auto X0 = graph->tensor(TensorAttr()
                                  .setName("in0")
                                  .setDim({n, c, h, w})
                                  .setStride({c * h * w, h * w, w, 1}));

      auto X1 = graph->tensor(TensorAttr()
                                  .setName("in1")
                                  .setDim(X0->getDim())
                                  .setStride(X0->getStride()));

      auto pointwiseAttr = PointwiseAttr().setMode(mode);
      auto pointwiseResult = graph->pointwise(X0, X1, pointwiseAttr);
      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      REQUIRE(isOk(graph->validate()));

      // Compile
      REQUIRE(isOk(graph->compile(handle, /*remove=*/true)));

      return std::make_tuple(graph, X0, X1, pointwiseResult);
    };

    Handle &handle = *handlePtr;
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, X0, X1, Y] = build_new_graph(handle);

    // Allocate first input buffer.
    auto x0Buf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle,
                         /*shape=*/castToSizeT(X0->getPhysicalDim()),
                         /*data=*/std::vector<T>(X0->getVolume(), x0))));

    // Allocate second input buffer.
    auto x1Buf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle,
                         /*shape=*/castToSizeT(X1->getPhysicalDim()),
                         /*data=*/std::vector<T>(X1->getVolume(), x1))));

    // Allocate output buffer.
    auto yBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle,
                         /*shape=*/castToSizeT(Y->getPhysicalDim()),
                         /*data=*/std::vector<T>(Y->getVolume(), T(0.0f)))));

    // Create variant pack.
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {X0, x0Buf},
            {X1, x1Buf},
            {Y, yBuf},
        };

    // Execute graph once.
    REQUIRE(isOk(graph->execute(variantPack)));

    // Calculate reference
    T y = 0;
    switch (mode) {
    case PointwiseAttr::Mode::ADD: {
      y = x0 + x1;
      break;
    }
    case PointwiseAttr::Mode::DIV: {
      y = x0 / x1;
      break;
    }
    case PointwiseAttr::Mode::MUL: {
      y = x0 * x1;
      break;
    }
    case PointwiseAttr::Mode::SUB: {
      y = x0 - x1;
      break;
    }
    default:
      FAIL("Unsupported pointwise mode: " << PointwiseAttr::modeToStr.at(mode));
    }

    // Read output buffers.
    std::vector<T> result;
    REQUIRE(isOk(yBuf->read(handle, result)));
    for (auto val : result)
      REQUIRE(val == y);

    // Execute graph a few times.
    constexpr size_t numIters = 1;
    for (size_t i = 0; i < numIters; i++)
      REQUIRE(isOk(graph->execute(variantPack)));

    // Repeat output buffer checks.
    result.clear();
    REQUIRE(isOk(yBuf->read(handle, result)));
    for (auto val : result)
      REQUIRE(val == y);
  };

  auto mode = GENERATE(PointwiseAttr::Mode::ADD, PointwiseAttr::Mode::DIV,
                       PointwiseAttr::Mode::MUL, PointwiseAttr::Mode::SUB);

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

  // fp32
  execute(handlePtr, mode, DataType::Float, float(-100.5), float(-20));
  // int32
  execute(handlePtr, mode, DataType::Int32, int(-50), int(-10));
  // fp16
  execute(handlePtr, mode, DataType::Half, half(-32.5), half(2));
  // int16
  execute(handlePtr, mode, DataType::Int16, int16_t(-5), int16_t(-2));
  // int8
  execute(handlePtr, mode, DataType::Int8, int8_t(-7), int8_t(2));
}
