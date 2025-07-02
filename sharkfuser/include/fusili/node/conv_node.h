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
  ConvFPropAttr attributes;

  ConvFPropNode(ConvFPropAttr &&attributes_, Context const &ctx)
      : NodeCRTP(ctx), attributes(std::move(attributes_)) {}

  Type getType() override final { return Type::CONVOLUTION; }

  error_t pre_validate_node() const override final {
    FUSILI_LOG_LABEL_ENDL("INFO: Validating node Type::Convolution "
                          << attributes.name << "...");
    FUSILI_RETURN_ERROR_IF(attributes.get_pre_padding().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv pre-padding not set");
    FUSILI_RETURN_ERROR_IF(attributes.get_post_padding().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv post-padding not set");
    FUSILI_RETURN_ERROR_IF(attributes.get_stride().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv stride not set");
    FUSILI_RETURN_ERROR_IF(attributes.get_dilation().empty(),
                           error_code_t::ATTRIBUTE_NOT_SET,
                           "Conv dilation not set");
    return {error_code_t::OK, ""};
  }

  error_t infer_properties_node() override final {
    FUSILI_LOG_LABEL_ENDL(
        "INFO: Inferring properties for node Type::Convolution "
        << attributes.name << "...");

    attributes.fill_from_context(context);

    auto &X = attributes.inputs.find(ConvFPropAttr::input_names::X)->second;
    auto &W = attributes.inputs.find(ConvFPropAttr::input_names::W)->second;
    auto &Y = attributes.outputs.find(ConvFPropAttr::output_names::Y)->second;

    return {error_code_t::OK, ""};
  }
};

} // namespace fusili
