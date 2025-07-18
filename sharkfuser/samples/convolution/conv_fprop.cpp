// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>

using namespace fusili;

TEST_CASE("Convolution fprop", "[conv][graph]") {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto graph = std::make_shared<Graph>();
  graph->set_io_data_type(DataType_t::HALF)
      .set_compute_data_type(DataType_t::FLOAT);

  auto X = graph->tensor(TensorAttr()
                             .set_name("image")
                             .set_dim({n, c, h, w})
                             .set_stride({c * h * w, h * w, w, 1}));

  auto W = graph->tensor(TensorAttr()
                             .set_name("filter")
                             .set_dim({k, c, r, s})
                             .set_stride({c * r * s, r * s, s, 1}));

  auto conv_attr = ConvFPropAttr()
                       .set_padding({0, 0})
                       .set_stride({1, 1})
                       .set_dilation({1, 1})
                       .set_name("conv_fprop");

  auto Y = graph->conv_fprop(X, W, conv_attr);

  // Specify Y's dimensions and strides
  Y->set_dim({n, k, h, w}).set_stride({k * h * w, h * w, w, 1});
  Y->set_output(true);

  REQUIRE(graph->validate().is_ok());
}
