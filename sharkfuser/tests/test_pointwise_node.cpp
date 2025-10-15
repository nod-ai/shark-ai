// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("PointwiseNode getName correctly propagates the attribute name",
          "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;
  attr.setName("foo_pointwise");

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_pointwise");
}

TEST_CASE("PointwiseNode getType returns correct type", "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::RELU_FWD);

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::Pointwise);
}

TEST_CASE("PointwiseNode preValidateNode detects missing mode",
          "[pointwise_node]") {
  Context ctx;

  SECTION("Mode not set") {
    PointwiseAttr attr;
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Pointwise mode not set");
  }

  SECTION("Mode set to RELU_FWD without inputs") {
    PointwiseAttr attr;
    attr.setMode(PointwiseAttr::Mode::RELU_FWD);
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "RELU_FWD mode requires IN_0 input");
  }

  SECTION("Mode set to ADD without second input") {
    PointwiseAttr attr;
    attr.setMode(PointwiseAttr::Mode::ADD);
    auto in0 = std::make_shared<TensorAttr>(1.0f);
    attr.setIN_0(in0);
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ADD mode requires IN_1 input");
  }

  SECTION("Mode set to RELU_FWD with too many inputs") {
    PointwiseAttr attr;
    attr.setMode(PointwiseAttr::Mode::RELU_FWD);
    auto in0 = std::make_shared<TensorAttr>(1.0f);
    auto in1 = std::make_shared<TensorAttr>(2.0f);
    attr.setIN_0(in0).setIN_1(in1);
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "RELU_FWD mode should not have IN_1 input set");
  }

  SECTION("Mode set to ADD with too many inputs") {
    PointwiseAttr attr;
    attr.setMode(PointwiseAttr::Mode::ADD);
    auto in0 = std::make_shared<TensorAttr>(1.0f);
    auto in1 = std::make_shared<TensorAttr>(2.0f);
    auto in2 = std::make_shared<TensorAttr>(3.0f);
    attr.setIN_0(in0).setIN_1(in1).setIN_2(in2);
    PointwiseNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "ADD mode should not have IN_2 input set");
  }
}

TEST_CASE("PointwiseNode with tensor attributes", "[pointwise_node]") {
  Context ctx;
  PointwiseAttr attr;

  attr.setMode(PointwiseAttr::Mode::RELU_FWD);

  auto in0 = std::make_shared<TensorAttr>(1.0f);
  auto in1 = std::make_shared<TensorAttr>(2.0f);
  auto out = std::make_shared<TensorAttr>(3.0f);

  attr.setIN_0(in0).setIN_1(in1).setOUT_0(out);

  PointwiseNode node(std::move(attr), ctx);

  // Verify the node has access to the attributes
  REQUIRE(node.pointwiseAttr.getIN_0() == in0);
  REQUIRE(node.pointwiseAttr.getIN_1() == in1);
  REQUIRE(node.pointwiseAttr.getOUT_0() == out);
  REQUIRE(node.pointwiseAttr.getMode() == PointwiseAttr::Mode::RELU_FWD);

  // Verify tensor properties
  REQUIRE(node.pointwiseAttr.getIN_0()->getDataType() == DataType::Float);
  REQUIRE(node.pointwiseAttr.getIN_1()->getDataType() == DataType::Float);
  REQUIRE(node.pointwiseAttr.getOUT_0()->getDataType() == DataType::Float);

  REQUIRE(node.pointwiseAttr.getIN_0()->getDim() == std::vector<int64_t>{1});
  REQUIRE(node.pointwiseAttr.getIN_1()->getDim() == std::vector<int64_t>{1});
  REQUIRE(node.pointwiseAttr.getOUT_0()->getDim() == std::vector<int64_t>{1});
}

TEST_CASE("PointwiseNode with ADD mode", "[pointwise_node]") {
  Context ctx;
  ctx.setIODataType(DataType::Float);
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::ADD);

  auto in0 = std::make_shared<TensorAttr>(1.0f);
  auto in1 = std::make_shared<TensorAttr>(2.0f);
  auto out = std::make_shared<TensorAttr>();

  attr.setIN_0(in0).setIN_1(in1).setOUT_0(out);

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.preValidateNode()));
  REQUIRE(isOk(node.inferPropertiesNode()));

  out = node.pointwiseAttr.getOUT_0();
  REQUIRE(out != nullptr);
  REQUIRE(out->getDim() == std::vector<int64_t>{1});
  REQUIRE(out->getDataType() == DataType::Float);
  REQUIRE(out->getStride() == std::vector<int64_t>{1});
}

TEST_CASE("PointwiseNode with ADD mode broadcast", "[pointwise_node]") {
  Context ctx;
  ctx.setIODataType(DataType::Float);
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::ADD);

  int64_t dim0 = 16;
  int64_t dim1 = 32;
  int64_t dim2 = 64;
  int64_t dim3 = 128;

  auto in0 = std::make_shared<TensorAttr>();
  in0->setDim({dim0, dim1, dim2, dim3})
      .setStride({dim1 * dim2 * dim3, dim2 * dim3, dim3, 1});
  auto in1 = std::make_shared<TensorAttr>();
  in1->setDim({1, dim1, 1, 1}).setStride({dim1, 1, 1, 1});
  auto out = std::make_shared<TensorAttr>();

  attr.setIN_0(in0).setIN_1(in1).setOUT_0(out);

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.preValidateNode()));
  REQUIRE(isOk(node.inferPropertiesNode()));

  out = node.pointwiseAttr.getOUT_0();
  REQUIRE(out != nullptr);
  REQUIRE(out->getDim() == in0->getDim());
  REQUIRE(out->getDataType() == DataType::Float);
  REQUIRE(out->getStride() == in0->getStride());
}

TEST_CASE("PointwiseNode with ADD mode invalid broadcast", "[pointwise_node]") {
  Context ctx;
  ctx.setIODataType(DataType::Float);
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::ADD);

  int64_t dim0 = 16;
  int64_t dim1 = 32;

  auto in0 = std::make_shared<TensorAttr>();
  in0->setDim({dim0}).setStride({1});
  auto in1 = std::make_shared<TensorAttr>();
  in1->setDim({dim1}).setStride({1});
  auto out = std::make_shared<TensorAttr>();

  attr.setIN_0(in0).setIN_1(in1).setOUT_0(out);

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.preValidateNode()));
  ErrorObject status = node.inferPropertiesNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "Cannot broadcast two non unit dimensions");
}

TEST_CASE("PointwiseNode with ADD mode double broadcast", "[pointwise_node]") {
  Context ctx;
  ctx.setIODataType(DataType::Float);
  PointwiseAttr attr;
  attr.setMode(PointwiseAttr::Mode::ADD);

  int64_t dim0 = 16;
  int64_t dim1 = 32;

  auto in0 = std::make_shared<TensorAttr>();
  in0->setDim({dim0, 1}).setStride({1, dim0});
  auto in1 = std::make_shared<TensorAttr>();
  in1->setDim({1, dim1}).setStride({dim1, 1});

  auto out = std::make_shared<TensorAttr>();
  out->setIsVirtual(true);

  attr.setIN_0(in0).setIN_1(in1).setOUT_0(out);

  PointwiseNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.preValidateNode()));
  REQUIRE(isOk(node.inferPropertiesNode()));

  out = node.pointwiseAttr.getOUT_0();
  REQUIRE(out != nullptr);
  REQUIRE(out->getDim() == std::vector{dim0, dim1});
  REQUIRE(out->getStride() == in0->getStride());
}
