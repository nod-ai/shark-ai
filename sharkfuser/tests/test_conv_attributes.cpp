// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

using namespace fusili;

TEST_CASE("ConvFPropAttr default constructor", "[conv_fprop_attr]") {
  ConvFPropAttr attr;
  REQUIRE(attr.get_stride().empty());
  REQUIRE(attr.get_pre_padding().empty());
  REQUIRE(attr.get_post_padding().empty());
  REQUIRE(attr.get_dilation().empty());
}

TEST_CASE("ConvFPropAttr setters and getters", "[conv_fprop_attr]") {
  ConvFPropAttr attr;
  std::vector<int64_t> stride = {1, 2};
  std::vector<int64_t> pre_padding = {0, 1};
  std::vector<int64_t> post_padding = {1, 0};
  std::vector<int64_t> dilation = {1, 1};

  attr.set_stride(stride)
      .set_pre_padding(pre_padding)
      .set_post_padding(post_padding)
      .set_dilation(dilation);

  REQUIRE(attr.get_stride() == stride);
  REQUIRE(attr.get_pre_padding() == pre_padding);
  REQUIRE(attr.get_post_padding() == post_padding);
  REQUIRE(attr.get_dilation() == dilation);

  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());

  auto x = std::make_shared<TensorAttr>(1.0f);
  auto w = std::make_shared<TensorAttr>(2.0f);
  auto y = std::make_shared<TensorAttr>(3.0f);

  attr.set_X(x).set_W(w).set_Y(y);

  REQUIRE(attr.inputs.size() == 2);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.get_X() == x);
  REQUIRE(attr.get_W() == w);
  REQUIRE(attr.get_Y() == y);

  REQUIRE(attr.get_X()->get_data_type() == DataType_t::FLOAT);
  REQUIRE(attr.get_W()->get_data_type() == DataType_t::FLOAT);
  REQUIRE(attr.get_Y()->get_data_type() == DataType_t::FLOAT);

  REQUIRE(attr.get_X()->get_dim() == std::vector<int64_t>{1});
  REQUIRE(attr.get_W()->get_dim() == std::vector<int64_t>{1});
  REQUIRE(attr.get_Y()->get_dim() == std::vector<int64_t>{1});

  REQUIRE(attr.get_X()->get_stride() == std::vector<int64_t>{1});
  REQUIRE(attr.get_W()->get_stride() == std::vector<int64_t>{1});
  REQUIRE(attr.get_Y()->get_stride() == std::vector<int64_t>{1});

  REQUIRE(attr.get_X()->get_is_scalar() == true);
  REQUIRE(attr.get_W()->get_is_scalar() == true);
  REQUIRE(attr.get_Y()->get_is_scalar() == true);

  REQUIRE(attr.get_X()->get_is_virtual() == false);
  REQUIRE(attr.get_W()->get_is_virtual() == false);
  REQUIRE(attr.get_Y()->get_is_virtual() == false);
}
