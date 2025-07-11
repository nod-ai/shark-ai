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
  auto x = g.tensor(TensorAttr()
                        .set_name("X")
                        .set_dim({1, 3, 8, 8})
                        .set_stride({192, 1, 24, 3}));
  auto w = g.tensor(TensorAttr()
                        .set_name("W")
                        .set_dim({4, 3, 3, 3})
                        .set_stride({27, 1, 9, 3}));
  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});
  auto y = g.conv_fprop(x, w, attr);
  y->set_output(true);
  REQUIRE(y->get_name() == "conv_fprop_0::Y");
  REQUIRE(y->get_is_virtual() == false);
}

TEST_CASE("Graph validate() returns OK for valid graph", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .set_name("x")
                        .set_dim({1, 3, 8, 8})
                        .set_stride({192, 1, 24, 3}));
  auto w = g.tensor(TensorAttr()
                        .set_name("w")
                        .set_dim({4, 3, 3, 3})
                        .set_stride({27, 1, 9, 3}));
  ConvFPropAttr attr;
  attr.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1}).set_name(
      "conv");
  auto y = g.conv_fprop(x, w, attr);
  y->set_dim({1, 4, 8, 8}).set_stride({256, 1, 32, 4});
  REQUIRE(g.validate().is_ok());
}

TEST_CASE("Graph query_tensor_of_uid finds tensors by UID", "[graph]") {
  Graph g;
  auto x = g.tensor(TensorAttr()
                        .set_name("X")
                        .set_dim({1, 3, 8, 8})
                        .set_stride({192, 1, 24, 3}));
  auto w = g.tensor(TensorAttr()
                        .set_name("W")
                        .set_dim({4, 3, 3, 3})
                        .set_stride({27, 1, 9, 3}));

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
