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
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Pointwise add with transposed operand", "[pointwise][graph]") {
  const int64_t n = 3, m = 2;

  // clang-format off
  const std::vector<float> inputData = {
    1.0f, 2.0f,
    3.0f, 4.0f,
    5.0f, 6.0f
  };

  // Result of inputData + transpose(inputData)
  const std::vector<float> expectedResult = {
    2.0f, 6.0f,
    5.0f, 9.0f,
    8.0f, 12.0f
  };
  // clang-format on

  // Parameterize sample by backend and create device-specific handles
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

  auto buildNewGraph = [&](const Handle &handle) {
    // Create graph
    auto graph = std::make_shared<Graph>();
    graph->setName("pointwise_add_transposed");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    // Tensor A: contiguous nxm tensor (row-major)
    auto aT =
        graph->tensor(TensorAttr().setName("input_a").setDim({n, m}).setStride(
            {m, 1})); // Contiguous: stride={2, 1}

    // Tensor B: transposed nxm tensor
    // Logical dim={n, m}, but stored with transposed strides
    auto bT = graph->tensor(TensorAttr()
                                .setName("input_b_transposed")
                                .setDim({n, m})
                                .setStride({1, n})); // Transposed

    // Create Pointwise ADD op
    auto pointwiseAttr = PointwiseAttr()
                             .setMode(PointwiseAttr::Mode::ADD)
                             .setName("add_transposed");
    auto resultT = graph->pointwise(aT, bT, pointwiseAttr);

    resultT->setName("result").setOutput(true);

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, aT, bT, resultT);
  };

  Handle &handle = *handlePtr;
  // Build graph for the given handle (device), validate and compile it.
  auto [graph, aT, bT, resultT] = buildNewGraph(handle);

  // Allocate input buffers and initialize with input data
  auto aBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT(aT->getPhysicalDim()), inputData)));
  auto bBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT(bT->getPhysicalDim()), inputData)));

  // Allocate output buffer
  auto resultBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, resultT, DataType::Float, 0.0f));

  // Create variant pack
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {aT, aBuf},
          {bT, bBuf},
          {resultT, resultBuf},
      };

  // Execute graph
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffer and verify against expected result
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(resultBuf->read(handle, result));
  REQUIRE(result == expectedResult);
}
