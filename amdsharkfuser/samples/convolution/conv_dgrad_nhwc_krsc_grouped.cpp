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

TEST_CASE("Convolution dgrad; DY/W (NHWC/KRSC), DX (NHWC); 1x1; no padding;"
          "grouped",
          "[conv][graph]") {
  constexpr int64_t n = 4, c = 16, h = 8, w = 8, k = 32, fc = 4, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_dgrad_sample_nhwc_krsc_1x1_nopad_grouped");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto dyT = graph->tensor(TensorAttr()
                                 .setName("dy")
                                 .setDim({n, k, h, w})
                                 .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto wT = graph->tensor(TensorAttr()
                                .setName("w")
                                .setDim({k, fc, r, s})
                                .setStride({fc * r * s, r * s, s, 1})); // KRSC

    auto dgradAttr = ConvDGradAttr()
                         .setStride({1, 1})
                         .setPadding({0, 0})
                         .setDilation({1, 1})
                         .setName("conv_dgrad");

    auto dxT = graph->convDGrad(dyT, wT, dgradAttr);
    dxT->setName("dx")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({n, c, h, w});

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, wT, dxT);
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

  auto [graph, dyT, wT, dxT] = buildNewGraph(handle);

  // Allocate input buffers.
  // Use values of 1.0 so the resulting DX for 1x1 conv equals k.
  const float inputScalar = 1.0f;
  auto dyBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, dyT, DataType::Float, inputScalar));
  auto wBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, wT, DataType::Float, inputScalar));
  auto dxBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, dxT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {wT, wBuf},
          {dxT, dxBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  std::vector<float> dxVals;
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));

  const float expected =
      (k / (c / static_cast<float>(fc))) * inputScalar * inputScalar;
  for (auto val : dxVals)
    REQUIRE(val == expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  dxVals.clear();
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));
  for (auto val : dxVals)
    REQUIRE(val == expected);
}
