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

// TODO(iree-org/iree#22405): This test is mark as "shouldfail" due to incorrect
// lowering of not unit-stide Grouped ConvWGrad in IREE. Please remove this tag
// when IREE supports this case.
TEST_CASE("Convolution wgrad; DY/X (NHWC), DW (KRSC); 1x1; no padding; grouped",
          "[conv][graph][!shouldfail]") {
  constexpr int64_t n = 4, c = 16, h = 8, w = 8, k = 32, fc = 4, r = 1, s = 1;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_wgrad_sample_nhwc_krsc_1x1_nopad_grouped");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto dyT = graph->tensor(TensorAttr()
                                 .setName("dy")
                                 .setDim({n, k, h, w})
                                 .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto wgradAttr = ConvWGradAttr()
                         .setStride({1, 1})
                         .setPadding({0, 0})
                         .setDilation({1, 1})
                         .setName("conv_wgrad");

    auto dwT = graph->convWGrad(dyT, xT, wgradAttr);
    dwT->setName("dw")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({k, fc, r, s});

    // Validate, infer missing properties
    FUSILLI_REQUIRE_OK(graph->validate());

    // Compile
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, xT, dwT);
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

  auto [graph, dyT, xT, dwT] = buildNewGraph(handle);

  // Allocate input buffers.
  constexpr float inputScalar = 1.0f;
  auto dyBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, dyT, DataType::Float, inputScalar));
  auto xBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, xT, DataType::Float, inputScalar));
  auto dwBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, dwT, DataType::Float, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {dwT, dwBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Read output buffer and validate values for 1x1, stride=1, no padding.
  std::vector<float> dwVals;
  FUSILLI_REQUIRE_OK(dwBuf->read(handle, dwVals));

  // Calculate expected output value.
  constexpr float expected =
      static_cast<float>(n * h * w) * inputScalar * inputScalar;
  for (auto val : dwVals)
    REQUIRE(val == expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

  // Repeat output buffer checks.
  dwVals.clear();
  FUSILLI_REQUIRE_OK(dwBuf->read(handle, dwVals));
  for (auto val : dwVals)
    REQUIRE(val == expected);
}
