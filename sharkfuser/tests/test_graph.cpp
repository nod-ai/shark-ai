// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace fusilli;

TEST_CASE("Graph getName correctly propagates the context name", "[graph]") {
  Graph g;
  g.setName("foo_graph");
  REQUIRE(g.getName() == "foo_graph");
}

TEST_CASE("Graph tensor() adds input tensor", "[graph]") {
  Graph g;
  auto t =
      g.tensor(TensorAttr().setName("input").setDim({2, 2}).setStride({2, 1}));
  REQUIRE(t->getName() == "input");
  REQUIRE(t->getDim() == std::vector<int64_t>({2, 2}));
  REQUIRE(t->getStride() == std::vector<int64_t>({2, 1}));
}

TEST_CASE("Graph conv_fprop() adds ConvFPropNode and output tensor",
          "[graph]") {
  Graph g;
  auto x =
      g.tensor(TensorAttr().setDim({1, 8, 8, 3}).setStride({192, 24, 3, 1}));
  auto w = g.tensor(TensorAttr().setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
  auto y = g.convFProp(x, w, attr);

  // Names for inputs are auto-populated when not set
  REQUIRE(x->getName() == "conv_fprop_0_X");
  REQUIRE(w->getName() == "conv_fprop_0_W");
  REQUIRE(y->getName() == "conv_fprop_0_Y");

  // Y is virtual (intermediate tensor) unless specified as output
  REQUIRE(y->isVirtual() == true);
  y->setOutput(true);
  REQUIRE(y->isVirtual() == false);
}

TEST_CASE("Graph validate() returns OK for valid graph", "[graph]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName("conv");
  auto y = g.convFProp(x, w, attr);

  // Fails because y is underspecified (shape/stride inference unimplemented)
  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  REQUIRE(status.getMessage() ==
          "ConvFProp node shape inference not implemented yet; please "
          "specify output tensor dimensions");

  // Specify y's shape and strides
  y->setDim({1, 8, 8, 4}).setStride({256, 32, 4, 1});
  REQUIRE(isOk(g.validate()));
}

TEST_CASE("Graph asm_emitter requires validation to be run first", "[graph]") {
  Graph g;
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName("conv");
  auto y = g.convFProp(x, w, attr);
  y->setDim({1, 8, 8, 4}).setStride({256, 32, 4, 1});

  // ASM emitter without validation should throw an error
  REQUIRE(isError(g.emitAsm()));
  // Validate the graph first
  REQUIRE(isOk(g.validate()));
  // ASM emitter should now work
  REQUIRE(isOk(g.emitAsm()));
}
