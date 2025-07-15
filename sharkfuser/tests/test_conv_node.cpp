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
  REQUIRE(node.pre_validate_node() == error_code_t::ATTRIBUTE_NOT_SET);
}

TEST_CASE("ConvFPropNode pre_validate_node passes with all attributes set",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.set_pre_padding({0, 0})
      .set_post_padding({0, 0})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.pre_validate_node().is_ok());
}

TEST_CASE("ConvFPropNode infer_properties_node returns NOT_IMPLEMENTED when Y "
          "is under specified",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.set_pre_padding({0, 0})
      .set_post_padding({0, 0})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  attr.set_X(std::make_shared<TensorAttr>(1.0f))
      .set_W(std::make_shared<TensorAttr>(2.0f))
      // Y is under specified (dim/stride missing)
      .set_Y(std::make_shared<TensorAttr>());

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.infer_properties_node() == error_code_t::NOT_IMPLEMENTED);
}

TEST_CASE(
    "ConvFPropNode infer_properties_node returns OK when Y is fully specified",
    "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  attr.set_pre_padding({0, 0})
      .set_post_padding({0, 0})
      .set_stride({1, 1})
      .set_dilation({1, 1});

  attr.set_X(std::make_shared<TensorAttr>(1.0f))
      .set_W(std::make_shared<TensorAttr>(2.0f))
      // Y is fully specified (dim/stride for scalar defaults to {1})
      .set_Y(std::make_shared<TensorAttr>(3.0f));

  ConvFPropNode node(std::move(attr), ctx);
  REQUIRE(node.infer_properties_node().is_ok());
}
