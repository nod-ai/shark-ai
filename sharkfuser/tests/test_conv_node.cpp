// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

using namespace fusili;

TEST_CASE("ConvFPropNode pre_validate_node detects missing attributes",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  // Leave attributes empty to trigger errors
  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.preValidateNode() == ErrorCode::AttributeNotSet);
}

TEST_CASE("ConvFPropNode preValidateNode passes with all attributes set",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.preValidateNode().isOk());
}

TEST_CASE("ConvFPropNode inferPropertiesNode returns NOT_IMPLEMENTED when Y "
          "is under specified",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  attr.setX(std::make_shared<TensorAttr>(1.0f))
      .setW(std::make_shared<TensorAttr>(2.0f))
      // Y is under specified (dim/stride missing)
      .setY(std::make_shared<TensorAttr>());

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.inferPropertiesNode() == ErrorCode::NotImplemented);
}

TEST_CASE(
    "ConvFPropNode inferPropertiesNode returns OK when Y is fully specified",
    "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  attr.setX(std::make_shared<TensorAttr>(1.0f))
      .setW(std::make_shared<TensorAttr>(2.0f))
      // Y is fully specified (dim/stride for scalar defaults to {1})
      .setY(std::make_shared<TensorAttr>(3.0f));

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.inferPropertiesNode().isOk());
}
