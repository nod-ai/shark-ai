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

TEST_CASE("ConvFPropNode getName correctly propagates the attribute name",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;
  attr.setName("foo_conv");

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_conv");
}

TEST_CASE("ConvFPropNode pre_validate_node detects missing attributes",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  // Leave attributes empty to trigger errors.
  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.preValidateNode() == ErrorCode::AttributeNotSet);
}

TEST_CASE("ConvFPropNode preValidateNode passes with all attributes set",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.preValidateNode()));
}

TEST_CASE("ConvFPropNode inferPropertiesNode (1D) when Y is fully specified",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  attr.setX(std::make_shared<TensorAttr>(1.0f))
      .setW(std::make_shared<TensorAttr>(2.0f))
      // Y is fully specified (dim/stride for scalar defaults to {1}).
      .setY(std::make_shared<TensorAttr>(3.0f));

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.inferPropertiesNode()));

  auto Y = node.convFPropAttr.getY();
  REQUIRE(Y->getDim() == std::vector<int64_t>{1});
  REQUIRE(Y->getStride() == std::vector<int64_t>{1});
}

TEST_CASE("ConvFPropNode inferPropertiesNode (1D) when Y is under specified",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  attr.setX(std::make_shared<TensorAttr>(1.0f))
      .setW(std::make_shared<TensorAttr>(2.0f))
      // Y is under specified (dim/stride missing).
      .setY(std::make_shared<TensorAttr>());

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.inferPropertiesNode()));

  auto Y = node.convFPropAttr.getY();
  REQUIRE(Y->getDim() == std::vector<int64_t>{1});
  REQUIRE(Y->getStride() == std::vector<int64_t>{1});
}

TEST_CASE("ConvFPropNode inferPropertiesNode (4D) when Y is under specified",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto X = std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1}));

  auto W = std::make_shared<TensorAttr>(
      TensorAttr().setDim({k, c, r, s}).setStride({c * r * s, r * s, s, 1}));

  attr.setX(X)
      .setW(W)
      // Y is under specified (dim/stride missing).
      .setY(std::make_shared<TensorAttr>());

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(isOk(node.inferPropertiesNode()));

  auto Y = node.convFPropAttr.getY();
  REQUIRE(Y->getDim() == std::vector<int64_t>({n, k, h, w}));
  REQUIRE(Y->getStride() == std::vector<int64_t>({k * h * w, h * w, w, 1}));
}
