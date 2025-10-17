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

TEST_CASE("Convolution wgrad; DY/X (NHWC), DW (KRSC); 1x1; no padding",
          "[conv][graph]") {
  int64_t n = 4, c = 8, h = 8, w = 8, k = 16, r = 1, s = 1;

  auto build_new_graph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("conv_wgrad_sample_nhwc_krsc_1x1_nopad");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto DY = graph->tensor(TensorAttr()
                                .setName("dy")
                                .setDim({n, k, h, w})
                                .setStride({k * h * w, 1, k * w, k})); // NHWC

    auto X = graph->tensor(TensorAttr()
                               .setName("x")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, 1, c * w, c})); // NHWC

    auto wgradAttr = ConvWGradAttr()
                         .setStride({1, 1})
                         .setPadding({0, 0})
                         .setDilation({1, 1})
                         .setName("conv_wgrad");

    auto DW = graph->convWGrad(DY, X, wgradAttr);
    DW->setName("dw")
        .setDataType(DataType::Float)
        .setOutput(true)
        .setDim({k, c, r, s});

    // Validate, infer missing properties
    REQUIRE(isOk(graph->validate()));

    // Compile
    REQUIRE(isOk(graph->compile(handle, /*remove=*/true)));

    return std::make_tuple(graph, DY, X, DW);
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

  auto [graph, DY, X, DW] = build_new_graph(handle);

  // Allocate input buffers.
  // Use values of 1.0 so the resulting DW for 1x1 conv equals N*H*W.
  const float inputScalar = 1.0f;
  auto dyBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(DY->getPhysicalDim()),
      /*data=*/std::vector<float>(DY->getVolume(), inputScalar))));

  auto xBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(X->getPhysicalDim()),
      /*data=*/std::vector<float>(X->getVolume(), inputScalar))));

  // Allocate output buffer (initialized to zeros).
  auto dwBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT(DW->getPhysicalDim()),
                       /*data=*/std::vector<float>(DW->getVolume(), 0.0f))));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {DY, dyBuf},
          {X, xBuf},
          {DW, dwBuf},
      };

  // Execute graph once.
  REQUIRE(isOk(graph->execute(variantPack)));

  // Read output buffer and validate values for 1x1, stride=1, no padding.
  std::vector<float> dwVals;
  REQUIRE(isOk(dwBuf->read(handle, dwVals)));

  const float expected =
      static_cast<float>(n * h * w) * inputScalar * inputScalar;
  for (auto val : dwVals)
    REQUIRE(val == expected);

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    REQUIRE(isOk(graph->execute(variantPack)));

  // Repeat output buffer checks.
  dwVals.clear();
  REQUIRE(isOk(dwBuf->read(handle, dwVals)));
  for (auto val : dwVals)
    REQUIRE(val == expected);
}
