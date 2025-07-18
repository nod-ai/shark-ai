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

  attr.setX(x).setW(w).setY(y);

  REQUIRE(attr.inputs.size() == 2);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.getX() == x);
  REQUIRE(attr.getW() == w);
  REQUIRE(attr.getY() == y);

  REQUIRE(attr.getX()->get_data_type() == DataType_t::FLOAT);
  REQUIRE(attr.getW()->get_data_type() == DataType_t::FLOAT);
  REQUIRE(attr.getY()->get_data_type() == DataType_t::FLOAT);

  REQUIRE(attr.getX()->get_dim() == std::vector<int64_t>{1});
  REQUIRE(attr.getW()->get_dim() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->get_dim() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->get_stride() == std::vector<int64_t>{1});
  REQUIRE(attr.getW()->get_stride() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->get_stride() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->get_is_scalar() == true);
  REQUIRE(attr.getW()->get_is_scalar() == true);
  REQUIRE(attr.getY()->get_is_scalar() == true);

  REQUIRE(attr.getX()->get_is_virtual() == false);
  REQUIRE(attr.getW()->get_is_virtual() == false);
  REQUIRE(attr.getY()->get_is_virtual() == false);
}
