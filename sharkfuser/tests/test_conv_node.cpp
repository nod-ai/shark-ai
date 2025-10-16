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

TEST_CASE("ConvFPropNode preValidateNode detects missing attributes",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  SECTION("Padding missing") {
    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Conv padding not set");
  }

  SECTION("Stride missing") {
    attr.setPadding({0});
    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Conv stride not set");
  }

  SECTION("Dilation missing") {
    attr.setPadding({0}).setStride({1});
    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Conv dilation not set");
  }

  SECTION("Input missing") {
    attr.setPadding({0}).setStride({1}).setDilation({1});
    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Conv input tensor X not set");
  }

  SECTION("Weight missing") {
    attr.setPadding({0}).setStride({1}).setDilation({1});
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Conv weight tensor W not set");
  }

  SECTION("All required attributes present") {
    attr.setPadding({0}).setStride({1}).setDilation({1});
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setW(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    ConvFPropNode node(std::move(attr), ctx);

    REQUIRE(isOk(node.preValidateNode()));
  }
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

TEST_CASE("ConvFPropNode preValidate checks on input stride validity",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto X = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({n, c, h, w})
                                            .setStride({c * h * w, 1, c * w, c})
                                            .setName("X_channels_last"));

  auto W = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({k, c, r, s})
                                            .setStride({c * r * s, c * s, 1, c})
                                            .setName("W_invalid_layout"));

  attr.setX(X).setW(W).setY(std::make_shared<TensorAttr>());

  ConvFPropNode node(std::move(attr), ctx);

  auto status = node.preValidateNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  REQUIRE(status.getMessage() ==
          "Tensor 'W_invalid_layout' is neither contiguous nor channels-last "
          "as defined by its stride");
}

TEST_CASE("ConvFPropNode postValidate checks on output stride validity",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto X = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({n, c, h, w})
                                            .setStride({c * h * w, h * w, w, 1})
                                            .setName("X_contig"));

  auto W = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({k, c, r, s})
                                            .setStride({c * r * s, 1, c * s, c})
                                            .setName("W_channels_last"));

  auto Y = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({n, k, h, w})
                                            .setStride({k * h * w, k * w, 1, k})
                                            .setName("Y_invalid_layout"));
  attr.setX(X).setW(W).setY(Y);

  ConvFPropNode node(std::move(attr), ctx);

  REQUIRE(isOk(node.preValidateNode()));
  REQUIRE(isOk(node.inferPropertiesNode()));

  auto status = node.postValidateNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  REQUIRE(status.getMessage() ==
          "Tensor 'Y_invalid_layout' is neither contiguous nor channels-last "
          "as defined by its stride");
}

TEST_CASE("ConvFPropNode rank checks", "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  int64_t n = 16, d = 2, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  SECTION("Input spatial dims check") {
    attr.setPadding({0}).setStride({1}).setDilation({1});

    auto X = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c}).setStride({c, 1}).setName("X_invalid"));

    auto W = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, c}).setStride({c, 1}).setName("W_invalid"));

    auto Y = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, k}).setStride({k, 1}).setName("Y_invalid"));
    attr.setX(X).setW(W).setY(Y);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv input tensor X must have a rank of at least 3");
  }

  SECTION("Output spatial dims check") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto X =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X_2d"));

    auto W =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("W_2d"));

    auto Y = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, k}).setStride({k, 1}).setName("Y_invalid"));
    attr.setX(X).setW(W).setY(Y);

    ConvFPropNode node(std::move(attr), ctx);

    REQUIRE(isOk(node.preValidateNode()));
    REQUIRE(isOk(node.inferPropertiesNode()));

    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv output tensor Y must have a rank of at least 3");
  }

  SECTION("Padding/stride/dilation rank check") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto X = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("X_3d"));

    auto W = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({k, c, d, r, s})
            .setStride({c * d * r * s, d * r * s, r * s, s, 1})
            .setName("W_3d"));

    auto Y = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("Y_3d"));
    attr.setX(X).setW(W).setY(Y);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv padding size does not match number of spatial dimensions");
  }

  SECTION("Input / weight rank check") {
    attr.setPadding({0, 0, 0}).setStride({1, 1, 1}).setDilation({1, 1, 1});

    auto X = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("X_3d"));

    auto W =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("W_2d"));

    auto Y = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("Y_3d"));
    attr.setX(X).setW(W).setY(Y);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv input tensor X and weight tensor W have different ranks");
  }

  SECTION("Input / output rank check") {
    attr.setPadding({0, 0, 0}).setStride({1, 1, 1}).setDilation({1, 1, 1});

    auto X = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("X_3d"));

    auto W = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({k, c, d, r, s})
            .setStride({c * d * r * s, d * r * s, r * s, s, 1})
            .setName("W_3d"));

    auto Y =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("Y_2d"));
    attr.setX(X).setW(W).setY(Y);

    ConvFPropNode node(std::move(attr), ctx);

    REQUIRE(isOk(node.preValidateNode()));
    REQUIRE(isOk(node.inferPropertiesNode()));

    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv input tensor X and output tensor Y have different ranks");
  }
}
