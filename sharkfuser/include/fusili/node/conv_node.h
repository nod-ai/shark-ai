// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_NODE_CONV_NODE_H
#define FUSILI_NODE_CONV_NODE_H

#include "fusili/attributes/conv_attributes.h"
#include "fusili/context.h"
#include "fusili/node/node.h"

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
      FUSILI_RETURN_ERROR_IF(true, error_code_t::NOT_IMPLEMENTED,
                             "Convolution node shape inference not implemented "
                             "yet; please specify output tensor dimensions");
    }

    if (y_t->get_stride().empty()) {
      FUSILI_RETURN_ERROR_IF(
          true, error_code_t::NOT_IMPLEMENTED,
          "Convolution node stride inference not implemented yet; please "
          "specify output tensor stride");
    }

    return {error_code_t::OK, ""};
  }
};

} // namespace fusili

#endif // FUSILI_NODE_CONV_NODE_H
