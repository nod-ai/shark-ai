// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <optional>

using namespace fusilli;

TEST_CASE("Convolution fprop", "[conv][graph]") {

  // Parameterize sample by backend
  std::optional<ErrorOr<FusilliHandle>> handle;
  SECTION("cpu backend") {
    handle.emplace(FusilliHandle::create(Backend::CPU));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("gfx942 backend") {
    handle.emplace(FusilliHandle::create(Backend::GFX942));
  }
#endif
  REQUIRE(handle.has_value());
  REQUIRE(isOk(*handle));

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto graph = std::make_shared<Graph>();
  graph->setName("fprop_sample");
  graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

  auto X = graph->tensor(TensorAttr()
                             .setName("image")
                             .setDim({n, c, h, w})
                             .setStride({c * h * w, h * w, w, 1}));

  auto W = graph->tensor(TensorAttr()
                             .setName("filter")
                             .setDim({k, c, r, s})
                             .setStride({c * r * s, r * s, s, 1}));

  auto conv_attr = ConvFPropAttr()
                       .setPadding({0, 0})
                       .setStride({1, 1})
                       .setDilation({1, 1})
                       .setName("conv_fprop");

  auto Y = graph->convFProp(X, W, conv_attr);

  // Specify Y's dimensions and strides
  Y->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
  Y->setOutput(true);

  REQUIRE(isOk(graph->validate()));

  REQUIRE(isOk(graph->compile(**handle, /*remove=*/true)));
}
