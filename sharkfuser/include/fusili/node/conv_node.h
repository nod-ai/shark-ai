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

  ConvFPropNode(ConvFPropAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), attr(std::move(attr)) {}

  Type getType() override final { return Type::Convolution; }

  error_t preValidateNode() const override final {
    FUSILI_LOG_LABEL_ENDL("INFO: Validating node Type::Convolution "
                          << attr.getName() << "...");
    FUSILI_RETURN_ERROR_IF(attr.getPrePadding().empty(),
                           error_code_t::AttributeNotSet,
                           "Conv pre-padding not set");
    FUSILI_RETURN_ERROR_IF(attr.getPostPadding().empty(),
                           error_code_t::AttributeNotSet,
                           "Conv post-padding not set");
    FUSILI_RETURN_ERROR_IF(attr.getStride().empty(),
                           error_code_t::AttributeNotSet,
                           "Conv stride not set");
    FUSILI_RETURN_ERROR_IF(attr.getDilation().empty(),
                           error_code_t::AttributeNotSet,
                           "Conv dilation not set");
    return {error_code_t::OK, ""};
  }

  error_t inferPropertiesNode() override final {
    FUSILI_LOG_LABEL_ENDL(
        "INFO: Inferring properties for node Type::Convolution "
        << attr.getName() << "...");

    attr.fillFromContext(context);

    // Default layouts for now
    auto xT = attr.getX(); // NHWC
    auto wT = attr.getW(); // KCRS
    auto yT = attr.getY(); // NKPQ

    const auto &xDim = xT->getDim();
    const auto &wDim = wT->getDim();
    const auto &yDim = yT->getDim();

    if (yDim.empty()) {
      FUSILI_RETURN_ERROR_IF(true, error_code_t::NotImplemented,
                             "Convolution node shape inference not implemented "
                             "yet; please specify output tensor dimensions");
    }

    if (yT->getStride().empty()) {
      FUSILI_RETURN_ERROR_IF(
          true, error_code_t::NotImplemented,
          "Convolution node stride inference not implemented yet; please "
          "specify output tensor stride");
    }

    return {error_code_t::OK, ""};
  }
};

} // namespace fusili

#endif // FUSILI_NODE_CONV_NODE_H
