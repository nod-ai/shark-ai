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
  auto t = g.tensor(
      TensorAttr().set_name("input").set_dim({2, 2}).set_stride({2, 1}));
  REQUIRE(t->get_name() == "input");
  REQUIRE(t->get_dim() == std::vector<int64_t>({2, 2}));
  REQUIRE(t->get_stride() == std::vector<int64_t>({2, 1}));
}

TEST_CASE("Graph conv_fprop() adds ConvFPropNode and output tensor",
          "[graph]") {
  Graph g;
  auto x =
      g.tensor(TensorAttr().set_dim({1, 8, 8, 3}).set_stride({192, 24, 3, 1}));
  auto w =
      g.tensor(TensorAttr().set_dim({4, 3, 3, 3}).set_stride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
  auto y = g.conv_fprop(x, w, attr);

  // Names for inputs are auto-populated when not set
  REQUIRE(x->get_name() == "conv_fprop_0::X");
  REQUIRE(w->get_name() == "conv_fprop_0::W");
  REQUIRE(y->get_name() == "conv_fprop_0::Y");

  // Y is virtual (intermediate tensor) unless specified as output
  REQUIRE(y->get_is_virtual() == true);
  y->set_output(true);
  REQUIRE(y->get_is_virtual() == false);
}

TEST_CASE("Graph validate() returns OK for valid graph", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .set_name("X")
                        .set_dim({1, 8, 8, 3})
                        .set_stride({192, 24, 3, 1}));
  auto w = g.tensor(TensorAttr()
                        .set_name("W")
                        .set_dim({4, 3, 3, 3})
                        .set_stride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}).set_name(
      "conv");
  auto y = g.conv_fprop(x, w, attr);

  // Fails because y is underspecified (shape/stride inference unimplemented)
  REQUIRE(g.validate().is_failure());

  // Specify y's shape and strides
  y->set_dim({1, 8, 8, 4}).set_stride({256, 32, 4, 1});
  REQUIRE(g.validate().is_ok());
}

TEST_CASE("Graph query_tensor_of_uid finds tensors by UID", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .set_name("X")
                        .set_dim({1, 8, 8, 3})
                        .set_stride({192, 24, 3, 1}));
  auto w = g.tensor(TensorAttr()
                        .set_name("W")
                        .set_dim({4, 3, 3, 3})
                        .set_stride({27, 9, 3, 1}));

  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}).set_name(
      "conv");
  auto y = g.conv_fprop(x, w, attr);
  y->set_output(true);

  x->set_uid(10);
  y->set_uid(20);

  TensorAttr found;
  REQUIRE(g.query_tensor_of_uid(10, found).is_ok());
  REQUIRE(found.get_name() == "X");
  REQUIRE(g.query_tensor_of_uid(20, found).is_ok());
  REQUIRE(found.get_name() == "conv::Y");
  REQUIRE(g.query_tensor_of_uid(999, found).is_failure());
}

TEST_CASE("Graph check for UID conflicts failing graph validation", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .set_name("X")
                        .set_dim({1, 8, 8, 3})
                        .set_stride({192, 24, 3, 1}));
  auto w = g.tensor(TensorAttr()
                        .set_name("W")
                        .set_dim({4, 3, 3, 3})
                        .set_stride({27, 9, 3, 1}));

  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}).set_name(
      "conv");
  auto y = g.conv_fprop(x, w, attr);
  y->set_dim({1, 8, 8, 4}).set_stride({256, 32, 4, 1});
  y->set_output(true);

  // Assign conflicting UIDs
  x->set_uid(42);
  w->set_uid(43);
  y->set_uid(42); // Conflict with x

  // Should fail validation due to UID conflict
  REQUIRE(g.validate().is_failure());

  // Assign unique UIDs
  y->set_uid(44);

  // Should pass validation now
  REQUIRE(g.validate().is_ok());
}
