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
  std::vector<int64_t> strideVec = {1, 2};
  std::vector<int64_t> paddingVec = {0, 1};
  std::vector<int64_t> dilationVec = {1, 1};

  std::span<int64_t> strideSpan(strideVec);
  std::span<int64_t> paddingSpan(paddingVec);
  std::span<int64_t> dilationSpan(dilationVec);

  // Setters either take a const std::vector & or a type constrained template,
  // std::span should call the templated override.
  attr.setStride(strideSpan).setPadding(paddingSpan).setDilation(dilationSpan);

  REQUIRE(attr.getStride() == strideVec);
  REQUIRE(attr.getPadding() == paddingVec);
  REQUIRE(attr.getDilation() == dilationVec);
}

TEST_CASE("ConvWGradAttr default constructor", "[conv_wgrad_attr]") {
  ConvWGradAttr attr;
  REQUIRE(attr.getStride().empty());
  REQUIRE(attr.getPadding().empty());
  REQUIRE(attr.getDilation().empty());
}

TEST_CASE("ConvWGradAttr setters and getters", "[conv_wgrad_attr]") {
  ConvWGradAttr attr;
  std::vector<int64_t> stride = {1, 2};
  std::vector<int64_t> padding = {0, 1};
  std::vector<int64_t> dilation = {1, 1};

  attr.setStride(stride).setPadding(padding).setDilation(dilation);

  REQUIRE(attr.getStride() == stride);
  REQUIRE(attr.getPadding() == padding);
  REQUIRE(attr.getDilation() == dilation);

  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());

  auto dy = std::make_shared<TensorAttr>(1.0f);
  auto x = std::make_shared<TensorAttr>(2.0f);
  auto dw = std::make_shared<TensorAttr>(3.0f);

  attr.setDY(dy).setX(x).setDW(dw);

  REQUIRE(attr.inputs.size() == 2);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.getDY() == dy);
  REQUIRE(attr.getX() == x);
  REQUIRE(attr.getDW() == dw);

  REQUIRE(attr.getDY()->getDataType() == DataType::Float);
  REQUIRE(attr.getX()->getDataType() == DataType::Float);
  REQUIRE(attr.getDW()->getDataType() == DataType::Float);

  REQUIRE(attr.getDY()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getX()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getDW()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getDY()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getX()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getDW()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getDY()->isScalar() == true);
  REQUIRE(attr.getX()->isScalar() == true);
  REQUIRE(attr.getDW()->isScalar() == true);

  REQUIRE(attr.getDY()->isVirtual() == false);
  REQUIRE(attr.getX()->isVirtual() == false);
  REQUIRE(attr.getDW()->isVirtual() == false);
}

TEST_CASE("ConvWGradAttr setter templated overrides", "[conv_wgrad_attr]") {
  ConvWGradAttr attr;
  std::vector<int64_t> strideVec = {1, 2};
  std::vector<int64_t> paddingVec = {0, 1};
  std::vector<int64_t> dilationVec = {1, 1};

  std::span<int64_t> strideSpan(strideVec);
  std::span<int64_t> paddingSpan(paddingVec);
  std::span<int64_t> dilationSpan(dilationVec);

  // Setters either take a const std::vector & or a type constrained template,
  // std::span should call the templated override.
  attr.setStride(strideSpan).setPadding(paddingSpan).setDilation(dilationSpan);

  REQUIRE(attr.getStride() == strideVec);
  REQUIRE(attr.getPadding() == paddingVec);
  REQUIRE(attr.getDilation() == dilationVec);
}

TEST_CASE("ConvDGradAttr default constructor", "[conv_dgrad_attr]") {
  ConvDGradAttr attr;
  REQUIRE(attr.getStride().empty());
  REQUIRE(attr.getPadding().empty());
  REQUIRE(attr.getDilation().empty());
}

TEST_CASE("ConvDGradAttr setters and getters", "[conv_dgrad_attr]") {
  ConvDGradAttr attr;
  std::vector<int64_t> stride = {1, 2};
  std::vector<int64_t> padding = {0, 1};
  std::vector<int64_t> dilation = {1, 1};

  attr.setStride(stride).setPadding(padding).setDilation(dilation);

  REQUIRE(attr.getStride() == stride);
  REQUIRE(attr.getPadding() == padding);
  REQUIRE(attr.getDilation() == dilation);

  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());

  auto dy = std::make_shared<TensorAttr>(1.0f);
  auto dx = std::make_shared<TensorAttr>(2.0f);
  auto w = std::make_shared<TensorAttr>(3.0f);

  attr.setDY(dy).setDX(dx).setW(w);

  REQUIRE(attr.inputs.size() == 2);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.getDY() == dy);
  REQUIRE(attr.getDX() == dx);
  REQUIRE(attr.getW() == w);

  REQUIRE(attr.getDY()->getDataType() == DataType::Float);
  REQUIRE(attr.getDX()->getDataType() == DataType::Float);
  REQUIRE(attr.getW()->getDataType() == DataType::Float);

  REQUIRE(attr.getDY()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getDX()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getW()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getDY()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getDX()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getW()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getDY()->isScalar() == true);
  REQUIRE(attr.getDX()->isScalar() == true);
  REQUIRE(attr.getW()->isScalar() == true);

  REQUIRE(attr.getDY()->isVirtual() == false);
  REQUIRE(attr.getDX()->isVirtual() == false);
  REQUIRE(attr.getW()->isVirtual() == false);
}

TEST_CASE("ConvDGradAttr setter templated overrides", "[conv_dgrad_attr]") {
  ConvDGradAttr attr;
  std::vector<int64_t> strideVec = {1, 2};
  std::vector<int64_t> paddingVec = {0, 1};
  std::vector<int64_t> dilationVec = {1, 1};

  std::span<int64_t> strideSpan(strideVec);
  std::span<int64_t> paddingSpan(paddingVec);
  std::span<int64_t> dilationSpan(dilationVec);

  // Setters either take a const std::vector & or a type constrained template,
  // std::span should call the templated override.
  attr.setStride(strideSpan).setPadding(paddingSpan).setDilation(dilationSpan);

  REQUIRE(attr.getStride() == strideVec);
  REQUIRE(attr.getPadding() == paddingVec);
  REQUIRE(attr.getDilation() == dilationVec);
}
