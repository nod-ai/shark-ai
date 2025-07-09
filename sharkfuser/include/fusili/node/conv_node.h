// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "../attributes/conv_attributes.h"
#include "../context.h"
#include "node.h"

namespace fusili {

class ConvFPropNode : public NodeCRTP<ConvFPropNode> {
public:
  ConvFPropAttr attr;

  ConvFPropNode(ConvFPropAttr &&attr_, Context const &ctx)
      : NodeCRTP(ctx), attr(std::move(attr_)) {}

  Type getType() override final { return Type::CONVOLUTION; }

  error_t pre_validate_node() const override final {
    FUSILI_LOG_LABEL_ENDL("INFO: Validating node Type::Convolution "
                          << attr.get_name() << "...");
    FUSILI_RETURN_ERROR_IF(attr.get_pre_padding().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv pre-padding not set");
    FUSILI_RETURN_ERROR_IF(attr.get_post_padding().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv post-padding not set");
    FUSILI_RETURN_ERROR_IF(attr.get_stride().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv stride not set");
    FUSILI_RETURN_ERROR_IF(attr.get_dilation().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv dilation not set");
    return {error_code_t::OK, ""};
  }

  error_t infer_properties_node() override final {
    FUSILI_LOG_LABEL_ENDL(
        "INFO: Inferring properties for node Type::Convolution "
        << attr.get_name() << "...");

    attr.fill_from_context(context);

    // Default layouts for now
    auto x_t = attr.get_X(); // NHWC
    auto w_t = attr.get_W(); // KCRS
    auto y_t = attr.get_Y(); // NKPQ

    auto const &x_dim = x_t->get_dim();
    auto const &w_dim = w_t->get_dim();
    auto const &y_dim = y_t->get_dim();

    if (y_dim.empty()) {
      FUSILI_RETURN_ERROR_IF(
          true, error_code_t::NOT_IMPLEMENTED,
          "Convolution node shape inference not implemented yet");

      // auto y_dim_inferred = std::vector<int64_t>(x_dim.size(), 1);

      // auto const &pre_padding = attr.get_pre_padding();
      // auto const &post_padding = attr.get_post_padding();
      // auto const &stride = attr.get_stride();
      // auto const &dilation = attr.get_dilation();

      // // N
      // y_dim_inferred[0] = x_dim[0]; // Batch size

      // // K
      // y_dim_inferred[1] = w_dim[0]; // Number of filters

      // // PQ
      // for (size_t dim = 2; dim < x_dim.size(); ++dim) {
      //   y_dim_inferred[dim] =
      //       1 + (x_dim[dim] - dilation[dim - 2] * (w_dim[dim] - 1) - 1 +
      //            pre_padding[dim - 2] + post_padding[dim - 2]) /
      //               stride[dim - 2];
      // }

      // y_t->set_dim(y_dim_inferred);
    }

    if (y_t->get_stride().empty()) {
      FUSILI_RETURN_ERROR_IF(
          true, error_code_t::NOT_IMPLEMENTED,
          "Convolution node stride inference not implemented yet");
    }

    return {error_code_t::OK, ""};
  }
};

} // namespace fusili
