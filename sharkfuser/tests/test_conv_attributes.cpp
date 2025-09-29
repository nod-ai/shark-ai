// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

using namespace fusilli;

TEST_CASE("ConvFPropAttr default constructor", "[conv_fprop_attr]") {
  ConvFPropAttr attr;
  REQUIRE(attr.getStride().empty());
  REQUIRE(attr.getPadding().empty());
  REQUIRE(attr.getDilation().empty());
}

TEST_CASE("ConvFPropAttr setters and getters", "[conv_fprop_attr]") {
  ConvFPropAttr attr;
  std::vector<int64_t> stride = {1, 2};
  std::vector<int64_t> padding = {0, 1};
  std::vector<int64_t> dilation = {1, 1};

  attr.setStride(stride).setPadding(padding).setDilation(dilation);

  REQUIRE(attr.getStride() == stride);
  REQUIRE(attr.getPadding() == padding);
  REQUIRE(attr.getDilation() == dilation);

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

  REQUIRE(attr.getX()->getDataType() == DataType::Float);
  REQUIRE(attr.getW()->getDataType() == DataType::Float);
  REQUIRE(attr.getY()->getDataType() == DataType::Float);

  REQUIRE(attr.getX()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getW()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getW()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getY()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getX()->isScalar() == true);
  REQUIRE(attr.getW()->isScalar() == true);
  REQUIRE(attr.getY()->isScalar() == true);

  REQUIRE(attr.getX()->isVirtual() == false);
  REQUIRE(attr.getW()->isVirtual() == false);
  REQUIRE(attr.getY()->isVirtual() == false);
}

TEST_CASE("ConvFPropAttr setter templated overrides", "[conv_fprop_attr]") {
  ConvFPropAttr attr;
  std::vector<int64_t> stride_vec = {1, 2};
  std::vector<int64_t> padding_vec = {0, 1};
  std::vector<int64_t> dilation_vec = {1, 1};

  std::span<int64_t> stride_span(stride_vec);
  std::span<int64_t> padding_span(padding_vec);
  std::span<int64_t> dilation_span(dilation_vec);

  // Setters either take a const std::vector & or a type constrained template,
  // std::span should call the templated override.
  attr.setStride(stride_span)
      .setPadding(padding_span)
      .setDilation(dilation_span);

  REQUIRE(attr.getStride() == stride_vec);
  REQUIRE(attr.getPadding() == padding_vec);
  REQUIRE(attr.getDilation() == dilation_vec);
}
