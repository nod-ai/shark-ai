// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusilli;

TEST_CASE("Multiple inputs use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  // First use of "arg0" is valid, second isn't
  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "Symbol name 'arg0' already in use");
}

TEST_CASE("Multiple outputs use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  auto y = g.convFProp(
      x, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv1"));
  // y's name is overridden to "result"
  y->setDim({1}).setStride({1}).setName("result");

  auto z = g.convFProp(
      y, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv2"));
  // z's name is also overridden to "result" which isn't valid
  z->setDim({1}).setStride({1}).setName("result");
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "Symbol name 'result' already in use");
}

TEST_CASE("Multiple outputs use same inferred name from producing nodes",
          "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  // This infers the name `conv_Y` (based on node name)
  auto y = g.convFProp(
      x, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv"));
  y->setDim({1}).setStride({1});

  // This also infers the name `conv_Y` (based on node name)
  auto z = g.convFProp(
      y, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv"));
  z->setDim({1}).setStride({1});
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "Symbol name 'conv_Y' already in use");
}

TEST_CASE("Multiple nodes use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  auto y = g.convFProp(
      x, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv"));
  // y is inferred to `conv_Y` based on node name
  y->setDim({1}).setStride({1});

  // Both conv nodes use the same name which is invalid as it'd break SSA
  // for the internal ops it'd generated (e.g. stride, padding etc).
  auto z = g.convFProp(
      y, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv"));
  z->setDim({1}).setStride({1}).setName("result");
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "Symbol name 'conv' already in use");
}

TEST_CASE("Input and outputs use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  auto y = g.convFProp(
      x, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv"));
  y->setDim({1}).setStride({1});

  auto z = g.convFProp(
      y, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv"));
  z->setDim({1}).setStride({1}).setName("arg0");
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "Symbol name 'arg0' already in use");
}

TEST_CASE("Input and nodes use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  auto y = g.convFProp(
      x, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "arg0"));
  y->setDim({1}).setStride({1});

  auto z = g.convFProp(
      y, w,
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv"));
  z->setDim({1}).setStride({1});
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() == "Symbol name 'arg0' already in use");
}

TEST_CASE("Unnamed graph with all names inferred", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setDim({1}).setStride({1}));

  auto y = g.convFProp(
      x, w, ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}));
  y->setDim({1}).setStride({1});

  auto z = g.convFProp(
      y, w, ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}));
  z->setDim({1}).setStride({1});
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(isOk(status));
  REQUIRE(x->getName() == "conv_fprop_0_X");
  REQUIRE(w->getName() == "conv_fprop_0_W");
  REQUIRE(y->getName() == "conv_fprop_0_Y");
  REQUIRE(z->getName() == "conv_fprop_1_Y");
}
