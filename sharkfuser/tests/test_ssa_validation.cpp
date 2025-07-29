// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusili;

TEST_CASE("Multiple inputs use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));

  auto status = g.validate();
  REQUIRE(status.isFailure());
  REQUIRE(status.getCode() == error_code_t::InvalidAttribute);
  REQUIRE(status.getMessage() == "Tensor with name 'arg0' already exists");
}

TEST_CASE("Multiple outputs use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  auto convAttr1 =
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv1");

  auto y = g.convFProp(x, w, convAttr1);
  y->setDim({1}).setStride({1}).setName("result");

  auto convAttr2 =
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv2");

  auto z = g.convFProp(y, w, convAttr2);
  z->setDim({1}).setStride({1}).setName("result");
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(status.isFailure());
  REQUIRE(status.getCode() == error_code_t::InvalidAttribute);
  REQUIRE(status.getMessage() == "Tensor with name 'result' already exists");
}

TEST_CASE("Multiple outputs use same inferred name from producing nodes",
          "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  auto convAttr1 =
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv");

  // This infers the name `conv_Y` (based on node name)
  auto y = g.convFProp(x, w, convAttr1);
  y->setDim({1}).setStride({1});

  auto convAttr2 =
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv");

  // This also infers the name `conv_Y` (based on node name)
  auto z = g.convFProp(y, w, convAttr2);
  z->setDim({1}).setStride({1});
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(status.isFailure());
  REQUIRE(status.getCode() == error_code_t::InvalidAttribute);
  REQUIRE(status.getMessage() == "Tensor with name 'conv_Y' already exists");
}

TEST_CASE("Multiple nodes use same name", "[graph][ssa]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto x = g.tensor(TensorAttr().setName("arg0").setDim({1}).setStride({1}));
  auto w = g.tensor(TensorAttr().setName("arg1").setDim({1}).setStride({1}));

  auto convAttr1 =
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv");

  auto y = g.convFProp(x, w, convAttr1);
  y->setDim({1}).setStride({1});

  auto convAttr2 =
      ConvFPropAttr().setPadding({0}).setStride({1}).setDilation({1}).setName(
          "conv");

  auto z = g.convFProp(y, w, convAttr2);
  z->setDim({1}).setStride({1}).setName("result");
  z->setOutput(true);

  auto status = g.validate();
  REQUIRE(status.isFailure());
  REQUIRE(status.getCode() == error_code_t::InvalidAttribute);
  REQUIRE(status.getMessage() == "Tensor with name 'conv_Y' already exists");
}
