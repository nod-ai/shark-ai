// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// Based on parameters, generates a unique name for the graph
static std::string generateName(PointwiseAttr::Mode mode, DataType type,
                                const std::vector<int64_t> &dim) {
  std::string name =
      std::format("pointwise_{}_dt{}_in0", PointwiseAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  for (const auto &d : dim) {
    name += std::format("_{}", d);
  }
  return name;
};

TEST_CASE("Pointwise unary ops", "[pointwise][graph]") {
  const auto dim = std::vector<int64_t>{2, 16, 64, 64};

  const auto mode = GENERATE(PointwiseAttr::Mode::RELU_FWD);

  auto execute = [&]<typename T>(const std::shared_ptr<Handle> &handlePtr,
                                 DataType dt, T x) {
    auto buildNewGraph = [&](const Handle &handle) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(mode, dt, dim));
      graph->setIODataType(dt).setComputeDataType(dt);

      // Initialize input tensors
      auto xT = graph->tensor(TensorAttr().setName("in0").setDim(dim).setStride(
          generateStrideFromDim(dim, getContiguousStrideOrder(dim.size()))));

      // Create Pointwise unary op
      auto pointwiseAttr = PointwiseAttr().setMode(mode);
      auto pointwiseResult = graph->pointwise(xT, pointwiseAttr);

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile
      FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

      return std::make_tuple(graph, xT, pointwiseResult);
    };

    Handle &handle = *handlePtr;
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, xT, yT] = buildNewGraph(handle);

    // Allocate input buffers.
    auto xBuf = FUSILLI_REQUIRE_UNWRAP(allocateBufferOfType(handle, xT, dt, x));

    // Allocate output buffer.
    auto yBuf =
        FUSILLI_REQUIRE_UNWRAP(allocateBufferOfType(handle, yT, dt, 0.0f));

    // Create variant pack.
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {xT, xBuf},
            {yT, yBuf},
        };

    // Execute graph once.
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

    // Calculate reference value
    T y = 0;
    switch (mode) {
    case PointwiseAttr::Mode::RELU_FWD: {
      y = std::max(x, T(0));
      break;
    }
    default:
      FAIL(
          "Unsupported pointwise mode: " << PointwiseAttr::kModeToStr.at(mode));
    }

    // Read output buffers.
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(val == y);

    // Execute graph a few times.
    constexpr size_t numIters = 1;
    for (size_t i = 0; i < numIters; i++)
      FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

    // Repeat output buffer checks.
    result.clear();
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (auto val : result)
      REQUIRE(val == y);
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

  // int32
  execute(handlePtr, DataType::Int32, int(-128));
  // fp16
  execute(handlePtr, DataType::Half, half(3.14));
}
