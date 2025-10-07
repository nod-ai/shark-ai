// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("PointwiseAttr default constructor", "[pointwise_attr]") {
  PointwiseAttr attr;
  REQUIRE(attr.getMode() == PointwiseAttr::Mode::NOT_SET);
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("PointwiseAttr setters and getters", "[pointwise_attr]") {
  PointwiseAttr attr;
  PointwiseAttr::Mode mode = PointwiseAttr::Mode::ADD;

  attr.setMode(mode);

  REQUIRE(attr.getMode() == mode);

  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());

  auto in0 = std::make_shared<TensorAttr>(1.0f);
  auto in1 = std::make_shared<TensorAttr>(2.0f);
  auto out = std::make_shared<TensorAttr>(3.0f);

  attr.setIN_0(in0).setIN_1(in1).setOUT_0(out);

  REQUIRE(attr.inputs.size() == 2);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.getIN_0() == in0);
  REQUIRE(attr.getIN_1() == in1);
  REQUIRE(attr.getOUT_0() == out);

  REQUIRE(attr.getIN_0()->getDataType() == DataType::Float);
  REQUIRE(attr.getIN_1()->getDataType() == DataType::Float);
  REQUIRE(attr.getOUT_0()->getDataType() == DataType::Float);

  REQUIRE(attr.getIN_0()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getIN_1()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getOUT_0()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getIN_0()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getIN_1()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getOUT_0()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getIN_0()->isScalar() == true);
  REQUIRE(attr.getIN_1()->isScalar() == true);
  REQUIRE(attr.getOUT_0()->isScalar() == true);

  REQUIRE(attr.getIN_0()->isVirtual() == false);
  REQUIRE(attr.getIN_1()->isVirtual() == false);
  REQUIRE(attr.getOUT_0()->isVirtual() == false);
}
