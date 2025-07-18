// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace fusili;

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
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
  auto y = g.conv_fprop(x, w, attr);

  // Names for inputs are auto-populated when not set
  REQUIRE(x->getName() == "conv_fprop_0::X");
  REQUIRE(w->getName() == "conv_fprop_0::W");
  REQUIRE(y->getName() == "conv_fprop_0::Y");

  // Y is virtual (intermediate tensor) unless specified as output
  REQUIRE(y->isVirtual() == true);
  y->setOutput(true);
  REQUIRE(y->isVirtual() == false);
}

TEST_CASE("Graph validate() returns OK for valid graph", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}).setName(
      "conv");
  auto y = g.conv_fprop(x, w, attr);

  // Fails because y is underspecified (shape/stride inference unimplemented)
  REQUIRE(g.validate().is_failure());

  // Specify y's shape and strides
  y->setDim({1, 8, 8, 4}).setStride({256, 32, 4, 1});
  REQUIRE(g.validate().is_ok());
}

TEST_CASE("Graph query_tensor_of_uid finds tensors by UID", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));

  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}).setName(
      "conv");
  auto y = g.conv_fprop(x, w, attr);
  y->setOutput(true);

  x->setUid(10);
  y->setUid(20);

  TensorAttr found;
  REQUIRE(g.query_tensor_of_uid(10, found).is_ok());
  REQUIRE(found.getName() == "X");
  REQUIRE(g.query_tensor_of_uid(20, found).is_ok());
  REQUIRE(found.getName() == "conv::Y");
  REQUIRE(g.query_tensor_of_uid(999, found).is_failure());
}

TEST_CASE("Graph check for UID conflicts failing graph validation", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));

  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}).setName(
      "conv");
  auto y = g.conv_fprop(x, w, attr);
  y->setDim({1, 8, 8, 4}).setStride({256, 32, 4, 1});
  y->setOutput(true);

  // Assign conflicting UIDs
  x->setUid(42);
  w->setUid(43);
  y->setUid(42); // Conflict with x

  // Should fail validation due to UID conflict
  REQUIRE(g.validate().is_failure());

  // Assign unique UIDs
  y->setUid(44);

  // Should pass validation now
  REQUIRE(g.validate().is_ok());
}
