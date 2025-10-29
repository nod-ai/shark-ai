// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>
#include <hip_utils.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Convolution fprop with hip stream; X (NCHW), W (KCRS); 1x1 conv; no "
          "padding",
          "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  // Get number of devices.
  int deviceCount;
  HIP_REQUIRE_SUCCESS(hipGetDeviceCount(&deviceCount));

  // Parameterize test by device id.
  int deviceId = 0;
  if (deviceCount > 1) {
    SECTION("non default device") {
      deviceId = deviceCount - 1;
      HIP_REQUIRE_SUCCESS(hipSetDevice(deviceId));
    }
  };

  // Create a HIP stream.
  hipStream_t stream;
  HIP_REQUIRE_SUCCESS(hipStreamCreate(&stream));

  Handle handle = FUSILLI_REQUIRE_UNWRAP(
      Handle::create(Backend::AMDGPU, /*deviceId=*/deviceId,
                     /*stream=*/reinterpret_cast<uintptr_t>(stream)));

  Graph graph;
  graph.setName("hip_stream_conv_fprop_sample_nchw_kcrs_1x1_nopad");
  graph.setIODataType(DataType::Half).setComputeDataType(DataType::Float);

  auto xT = graph.tensor(TensorAttr()
                             .setName("image")
                             .setDim({n, c, h, w})
                             .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto wT = graph.tensor(TensorAttr()
                             .setName("filter")
                             .setDim({k, c, r, s})
                             .setStride({c * r * s, r * s, s, 1})); // KCRS

  auto convAttr = ConvFPropAttr()
                      .setPadding({0, 0})
                      .setStride({1, 1})
                      .setDilation({1, 1})
                      .setName("conv_fprop");

  auto yT = graph.convFProp(xT, wT, convAttr);
  yT->setOutput(true);

  // Validate, infer missing properties
  FUSILLI_REQUIRE_OK(graph.validate());

  // Compile
  FUSILLI_REQUIRE_OK(graph.compile(handle, /*remove=*/true));

  // Allocate input buffer.
  auto xBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, xT, DataType::Half, 1.0f));

  // Allocate weight buffer.
  auto wBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, wT, DataType::Half, 1.0f));

  // Allocate output buffer.
  auto yBuf = FUSILLI_REQUIRE_UNWRAP(
      allocateBufferOfType(handle, yT, DataType::Half, 0.0f));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {wT, wBuf},
          {yT, yBuf},
      };

  // Execute graph once.
  FUSILLI_REQUIRE_OK(graph.execute(handle, variantPack));

  // Read output buffers.
  std::vector<half> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(128.0f));

  // Execute graph a few times.
  constexpr size_t numIters = 1;
  for (size_t i = 0; i < numIters; i++)
    FUSILLI_REQUIRE_OK(graph.execute(handle, variantPack));

  // Repeat output buffer checks.
  result.clear();
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
  for (auto val : result)
    REQUIRE(val == half(128.0f));
}
