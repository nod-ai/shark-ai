// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <utility>
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

  SECTION("Output missing") {
    attr.setPadding({0}).setStride({1}).setDilation({1});
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setW(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Conv output tensor Y not set");
  }

  SECTION("All required attributes present") {
    attr.setPadding({0}).setStride({1}).setDilation({1});
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setW(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    ConvFPropNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
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
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.convFPropAttr.getY();
  REQUIRE(yT->getDim() == std::vector<int64_t>{1});
  REQUIRE(yT->getStride() == std::vector<int64_t>{1});
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
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.convFPropAttr.getY();
  REQUIRE(yT->getDim() == std::vector<int64_t>{1});
  REQUIRE(yT->getStride() == std::vector<int64_t>{1});
}

TEST_CASE("ConvFPropNode inferPropertiesNode (4D) when Y is under specified",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto xT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c, h, w}).setStride({c * h * w, h * w, w, 1}));

  auto wT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({k, c, r, s}).setStride({c * r * s, r * s, s, 1}));

  attr.setX(xT)
      .setW(wT)
      // Y is under specified (dim/stride missing).
      .setY(std::make_shared<TensorAttr>());

  ConvFPropNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.convFPropAttr.getY();
  REQUIRE(yT->getDim() == std::vector<int64_t>({n, k, h, w}));
  REQUIRE(yT->getStride() == std::vector<int64_t>({k * h * w, h * w, w, 1}));
}

TEST_CASE("ConvFPropNode preValidate checks on input stride validity",
          "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto xT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, c, h, w})
                                       .setStride({c * h * w, 1, c * w, c})
                                       .setName("X_channels_last"));

  auto wT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({k, c, r, s})
                                       .setStride({c * r * s, c * s, 1, c})
                                       .setName("W_invalid_layout"));

  attr.setX(xT).setW(wT).setY(std::make_shared<TensorAttr>());

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

  auto xT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, c, h, w})
                                       .setStride({c * h * w, h * w, w, 1})
                                       .setName("X_contig"));

  auto wT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({k, c, r, s})
                                       .setStride({c * r * s, 1, c * s, c})
                                       .setName("W_channels_last"));

  auto yT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, k, h, w})
                                       .setStride({k * h * w, k * w, 1, k})
                                       .setName("Y_invalid_layout"));
  attr.setX(xT).setW(wT).setY(yT);

  ConvFPropNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

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

    auto xT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c}).setStride({c, 1}).setName("X_invalid"));

    auto wT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, c}).setStride({c, 1}).setName("W_invalid"));

    auto yT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, k}).setStride({k, 1}).setName("Y_invalid"));
    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv input tensor X must have a rank of at least 3");
  }

  SECTION("Output spatial dims check") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X_2d"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("W_2d"));

    auto yT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, k}).setStride({k, 1}).setName("Y_invalid"));
    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv output tensor Y must have a rank of at least 3");
  }

  SECTION("Padding/stride/dilation rank check") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto xT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("X_3d"));

    auto wT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({k, c, d, r, s})
            .setStride({c * d * r * s, d * r * s, r * s, s, 1})
            .setName("W_3d"));

    auto yT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("Y_3d"));
    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv padding size does not match number of spatial dimensions");
  }

  SECTION("Input / weight rank check") {
    attr.setPadding({0, 0, 0}).setStride({1, 1, 1}).setDilation({1, 1, 1});

    auto xT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("X_3d"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("W_2d"));

    auto yT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("Y_3d"));
    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv input tensor X and weight tensor W have different ranks");
  }

  SECTION("Input / output rank check") {
    attr.setPadding({0, 0, 0}).setStride({1, 1, 1}).setDilation({1, 1, 1});

    auto xT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("X_3d"));

    auto wT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({k, c, d, r, s})
            .setStride({c * d * r * s, d * r * s, r * s, s, 1})
            .setName("W_3d"));

    auto yT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("Y_2d"));
    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv input tensor X and output tensor Y have different ranks");
  }
}

TEST_CASE("ConvFPropNode group count checks", "[conv_node]") {
  Context ctx;
  ConvFPropAttr attr;

  int64_t n = 8, h = 16, w = 16, r = 1, s = 1;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  SECTION("Valid configuration of attributes") {
    int64_t c = 4, k = 8, fc = 2;

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto yT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("Y"));

    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Input channels must be divisible by the filter channels") {
    int64_t c = 6, k = 16, fc = 4;

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto yT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("Y"));

    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv input channels must be divisible by the filter channels");
  }

  SECTION("Output channels must be divisible by the filter channels") {
    int64_t c = 16, k = 25, fc = 4;

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto yT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("Y"));

    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv output channels must be divisible by the group count");
  }

  SECTION("Group count is in the correct range") {
    int64_t c = 32, k = 8, fc = 2;

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto yT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("Y"));

    attr.setX(xT).setW(wT).setY(yT);

    ConvFPropNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Conv group count must be greater than 0 and less than or equal to "
            "the numbers of input and outputs channels");
  }
}

TEST_CASE("ConvWGradNode preValidateNode detects missing attributes",
          "[conv_wgrad_node]") {
  Context ctx;
  ConvWGradAttr attr;

  SECTION("Padding missing") {
    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvWGrad padding not set");
  }

  SECTION("Stride missing") {
    attr.setPadding({0, 0});
    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvWGrad stride not set");
  }

  SECTION("Dilation missing") {
    attr.setPadding({0, 0}).setStride({1, 1});
    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvWGrad dilation not set");
  }

  SECTION("DY tensor missing") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvWGrad gradient tensor DY not set");
  }

  SECTION("X tensor missing") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
    attr.setDY(std::make_shared<TensorAttr>(1.0f));
    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvWGrad input tensor X not set");
  }

  SECTION("DW tensor missing") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
    attr.setDY(std::make_shared<TensorAttr>(1.0f));
    attr.setX(std::make_shared<TensorAttr>(2.0f));
    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvWGrad output tensor DW not set");
  }

  SECTION("All required attributes present") {
    attr.setPadding({0}).setStride({1}).setDilation({1});
    attr.setDY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setDW(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    ConvWGradNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }
}

TEST_CASE("ConvWGradNode preValidate checks on input stride validity",
          "[conv_wgrad_node]") {
  Context ctx;
  ConvWGradAttr attr;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto dyT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, k, h, w})
                                       .setStride({k * h * w, 1, k * w, k})
                                       .setName("DY_channels_last"));

  auto xT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, c, h, w})
                                       .setStride({c * h * w, c * w, 1, c})
                                       .setName("X_invalid_layout"));

  auto dwT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({k, c, r, s})
                                       .setStride({c * r * s, r * s, s, 1})
                                       .setName("DW"));

  attr.setDY(dyT).setX(xT).setDW(dwT);

  ConvWGradNode node(std::move(attr), ctx);

  auto status = node.preValidateNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  REQUIRE(status.getMessage() ==
          "Tensor 'X_invalid_layout' is neither contiguous nor channels-last "
          "as defined by its stride");
}

TEST_CASE("ConvWGradNode rank checks", "[conv_wgrad_node]") {
  Context ctx;
  ConvWGradAttr attr;

  int64_t n = 16, d = 2, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  SECTION("Input spatial dims check (DY/X rank >= 3)") {
    attr.setPadding({0}).setStride({1}).setDilation({1});

    auto dyT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, k}).setStride({k, 1}).setName("DY_invalid"));

    auto xT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c}).setStride({c, 1}).setName("X_invalid"));

    auto dwT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, c}).setStride({c, 1}).setName("DW_invalid"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "ConvWGrad input tensors DY/X must have a rank of at least 3");
  }

  SECTION("Output spatial dims check (DW rank >= 3)") {
    attr = ConvWGradAttr();
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("DY_2d"));

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X_2d"));

    auto dwT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, c}).setStride({c, 1}).setName("DW_invalid"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "ConvWGrad weight gradient tensor DW must have a rank of at least 3");
  }

  SECTION("Padding/stride/dilation rank check match spatial dims") {
    attr = ConvWGradAttr();
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto dyT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("DY_3d"));

    auto xT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("X_3d"));

    auto dwT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("DW_3d"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "ConvWGrad padding size does not match number of spatial dimensions");
  }

  SECTION("DY / X rank check (must match)") {
    attr = ConvWGradAttr();
    attr.setPadding({0, 0, 0}).setStride({1, 1, 1}).setDilation({1, 1, 1});

    auto dyT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("DY_3d"));

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("X_2d"));

    auto dwT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("DW_3d"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "ConvWGrad tensors DY and X have different ranks");
  }
}

TEST_CASE("ConvWGradNode postValidateNode dimension validation",
          "[conv_wgrad_node]") {
  Context ctx;
  ConvWGradAttr attr;
  int64_t n = 16, c = 256, h = 64, w = 32, k = 128, r = 1, s = 1;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto dyT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, k, h, w})
                                       .setStride({k * h * w, 1, k * w, k})
                                       .setName("DY"));

  auto xT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, c, h, w})
                                       .setStride({c * h * w, 1, c * w, c})
                                       .setName("X"));

  // Wrong DW dimensions - should be {k, c, r, s} but using {c, k, r, s}
  auto dwT = std::make_shared<TensorAttr>(
      TensorAttr()
          .setDim({c, k, r, s}) // Wrong order: c, k instead of k, c
          .setStride({k * r * s, r * s, s, 1})
          .setName("DW"));

  attr.setDY(dyT).setX(xT).setDW(dwT);

  ConvWGradNode node(std::move(attr), ctx);

  // First pass pre-validation
  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  // Post-validation should fail due to incorrect dimensions
  auto postStatus = node.postValidateNode();
  REQUIRE(isError(postStatus));
  REQUIRE(postStatus.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(postStatus.getMessage() ==
          "ConvWGrad output DW dimensions do not match the expected shapes "
          "inferred based on input dimensions");
}

TEST_CASE("ConvWGradNode group count checks", "[conv_wgrad_node]") {
  Context ctx;
  ConvWGradAttr attr;

  int64_t n = 8, h = 16, w = 16, r = 1, s = 1;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  SECTION("Valid configuration of attributes") {
    int64_t c = 4, k = 8, fc = 2;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, 1, k * w, k})
                                         .setName("DY"));

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, 1, c * w, c})
                                         .setName("X"));

    auto dwT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("DW"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Input channels must be divisible by the filter channels") {
    int64_t c = 6, k = 16, fc = 4;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, 1, k * w, k})
                                         .setName("DY"));

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, 1, c * w, c})
                                         .setName("X"));

    auto dwT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("DW"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "ConvWGrad input (X) channels must be "
                                   "divisible by the filter (W) channels");
  }

  SECTION("Output channels must be divisible by the filter channels") {
    int64_t c = 16, k = 25, fc = 4;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, 1, k * w, k})
                                         .setName("DY"));

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, 1, c * w, c})
                                         .setName("X"));

    auto dwT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("DW"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "ConvWGrad output (DY) channels must be divisible by the group count");
  }

  SECTION("Group count is in the correct range") {
    int64_t c = 32, k = 8, fc = 2;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, 1, k * w, k})
                                         .setName("DY"));

    auto xT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, 1, c * w, c})
                                         .setName("X"));

    auto dwT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("DW"));

    attr.setDY(dyT).setX(xT).setDW(dwT);

    ConvWGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "ConvWGrad group count must be greater than "
                                   "0 and less than or equal to "
                                   "the numbers of input and outputs channels");
  }
}

TEST_CASE("ConvDGradNode preValidateNode detects missing attributes",
          "[conv_dgrad_node]") {
  Context ctx;
  ConvDGradAttr attr;

  SECTION("Padding missing") {
    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvDGrad padding not set");
  }

  SECTION("Stride missing") {
    attr.setPadding({0, 0});
    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvDGrad stride not set");
  }

  SECTION("Dilation missing") {
    attr.setPadding({0, 0}).setStride({1, 1});
    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvDGrad dilation not set");
  }

  SECTION("DY tensor missing") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvDGrad gradient tensor DY not set");
  }

  SECTION("W tensor missing") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
    attr.setDY(std::make_shared<TensorAttr>(1.0f));
    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvDGrad input tensor W not set");
  }

  SECTION("DX tensor missing") {
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
    attr.setDY(std::make_shared<TensorAttr>(1.0f));
    attr.setW(std::make_shared<TensorAttr>(1.0f));
    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "ConvDGrad output tensor DX not set");
  }

  SECTION("All required attributes present") {
    attr.setPadding({0}).setStride({1}).setDilation({1});
    attr.setDY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setW(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    attr.setDX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    ConvDGradNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }
}

TEST_CASE("ConvDGradNode preValidate checks on input stride validity",
          "[conv_dgrad_node]") {
  Context ctx;
  ConvDGradAttr attr;

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto dyT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, k, h, w})
                                       .setStride({k * h * w, 1, k * w, k})
                                       .setName("DY_channels_last"));
  auto wT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({k, c, r, s})
                                       .setStride({c * r * s, c * s, 1, c})
                                       .setName("W_invalid_layout"));
  auto dxT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, c, h, w})
                                       .setStride({c * h * w, c * w, 1, c})
                                       .setName("DX"));
  attr.setDY(dyT).setW(wT).setDX(dxT);

  ConvDGradNode node(std::move(attr), ctx);

  auto status = node.preValidateNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  REQUIRE(status.getMessage() ==
          "Tensor 'W_invalid_layout' is neither contiguous nor channels-last "
          "as defined by its stride");
}

TEST_CASE("ConvDGradNode rank checks", "[conv_dgrad_node]") {
  Context ctx;
  ConvDGradAttr attr;

  int64_t n = 16, d = 2, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  SECTION("Spatial dims check (DY/W rank >= 3)") {
    attr.setPadding({0}).setStride({1}).setDilation({1});

    auto dyT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, k}).setStride({k, 1}).setName("DY_invalid"));
    auto wT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, c}).setStride({c, 1}).setName("DW_invalid"));
    auto dxT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c}).setStride({c, 1}).setName("X_invalid"));
    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "ConvDGrad input tensors DY/W must have a rank of at least 3");
  }

  SECTION("Spatial dims check (W rank >= 3)") {
    attr = ConvDGradAttr();
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("DY_2d"));
    auto wT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, c}).setStride({c, 1}).setName("W_invalid"));
    auto dxT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("DX_2d"));
    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "ConvDGrad input tensors DY/W must have a rank of at least 3");
  }

  SECTION("Padding/stride/dilation rank check match spatial dims") {
    attr = ConvDGradAttr();
    attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

    auto dyT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("DY_3d"));
    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("W_3d"));
    auto dxT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, c, d, h, w})
            .setStride({c * d * h * w, d * h * w, h * w, w, 1})
            .setName("DX_3d"));
    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "ConvDGrad padding size does not match number of spatial dimensions");
  }

  SECTION("DY / DX rank check (must match)") {
    attr = ConvDGradAttr();
    attr.setPadding({0, 0, 0}).setStride({1, 1, 1}).setDilation({1, 1, 1});

    auto dyT = std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({n, k, d, h, w})
            .setStride({k * d * h * w, d * h * w, h * w, w, 1})
            .setName("DY_3d"));
    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, c, r, s})
                                         .setStride({c * r * s, r * s, s, 1})
                                         .setName("W_3d"));
    auto dxT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("DX_2d"));
    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "ConvDGrad tensors DY and DX have different ranks");
  }
}

TEST_CASE("ConvDGradNode postValidateNode dimension validation",
          "[conv_dgrad_node]") {
  Context ctx;
  ConvDGradAttr attr;
  int64_t n = 16, c = 256, h = 64, w = 32, k = 128, r = 1, s = 1;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  auto dyT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, k, h, w})
                                       .setStride({k * h * w, 1, k * w, k})
                                       .setName("DY"));
  // Wrong W dimensions - should be {k, c, r, s} but using {c, k, r, s}
  auto wT = std::make_shared<TensorAttr>(
      TensorAttr()
          .setDim({c, k, r, s}) // Wrong order: c, k instead of k, c
          .setStride({k * r * s, r * s, s, 1})
          .setName("W"));
  auto dxT =
      std::make_shared<TensorAttr>(TensorAttr()
                                       .setDim({n, c, h, w})
                                       .setStride({c * h * w, 1, c * w, c})
                                       .setName("DX"));

  attr.setDY(dyT).setW(wT).setDX(dxT);

  ConvDGradNode node(std::move(attr), ctx);

  // First pass pre-validation
  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  // Post-validation should fail due to incorrect dimensions
  auto postStatus = node.postValidateNode();
  REQUIRE(isError(postStatus));
  REQUIRE(postStatus.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(postStatus.getMessage() ==
          "ConvDGrad DY dimensions do not match the "
          "expected shapes inferred based on DX and W");
}

TEST_CASE("ConvDGradNode group count checks", "[conv_dgrad_node]") {
  Context ctx;
  ConvDGradAttr attr;

  int64_t n = 8, h = 16, w = 16, r = 1, s = 1;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});

  SECTION("Valid configuration of attributes") {
    int64_t c = 4, k = 8, fc = 2;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("DY"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto dxT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("DX"));

    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Input channels must be divisible by the filter channels") {
    int64_t c = 6, k = 16, fc = 4;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("DY"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto dxT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("DX"));

    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "ConvDGrad input (DX) channels must be "
                                   "divisible by the filter (W) channels");
  }

  SECTION("Output channels must be divisible by the filter channels") {
    int64_t c = 16, k = 25, fc = 4;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("DY"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto dxT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("DX"));

    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "ConvDGrad output (DY) channels must be divisible by the group count");
  }

  SECTION("Group count is in the correct range") {
    int64_t c = 32, k = 8, fc = 2;

    auto dyT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, k, h, w})
                                         .setStride({k * h * w, h * w, w, 1})
                                         .setName("DY"));

    auto wT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({k, fc, r, s})
                                         .setStride({fc * r * s, r * s, s, 1})
                                         .setName("W"));

    auto dxT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({n, c, h, w})
                                         .setStride({c * h * w, h * w, w, 1})
                                         .setName("DX"));

    attr.setDY(dyT).setW(wT).setDX(dxT);

    ConvDGradNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "ConvDGrad group count must be greater than 0 and less than or equal "
        "to the numbers of input and outputs channels");
  }
}
