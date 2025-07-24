// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the convolution nodes like
// `ConvFPropNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILI_NODE_CONV_NODE_H
#define FUSILI_NODE_CONV_NODE_H

#include "fusili/attributes/conv_attributes.h"
#include "fusili/context.h"
#include "fusili/node/node.h"

#include <string>

namespace fusili {

class ConvFPropNode : public NodeCRTP<ConvFPropNode> {
public:
  ConvFPropAttr convFPropAttr;

  ConvFPropNode(ConvFPropAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), convFPropAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const override final;
  std::string getOperandTypesAsm() const override final;
  std::string getResultNamesAsm() const override final;
  std::string getResultTypesAsm() const override final;
  std::string getStrideOpsAsm() const;
  std::string getPaddingOpsAsm() const;
  std::string getDilationOpsAsm() const;

  Type getType() override final { return Type::Convolution; }

  error_t preValidateNode() const override final {
    FUSILI_LOG_LABEL_ENDL("INFO: Validating node Type::Convolution "
                          << convFPropAttr.getName());
    FUSILI_RETURN_ERROR_IF(convFPropAttr.getPadding().empty(),
                           error_code_t::AttributeNotSet,
                           "Conv padding not set");
    FUSILI_RETURN_ERROR_IF(convFPropAttr.getStride().empty(),
                           error_code_t::AttributeNotSet,
                           "Conv stride not set");
    FUSILI_RETURN_ERROR_IF(convFPropAttr.getDilation().empty(),
                           error_code_t::AttributeNotSet,
                           "Conv dilation not set");
    return {error_code_t::OK, ""};
  }

  error_t inferPropertiesNode() override final {
    FUSILI_LOG_LABEL_ENDL(
        "INFO: Inferring properties for node Type::Convolution "
        << convFPropAttr.getName());

    convFPropAttr.fillFromContext(context);

    // Default layouts for now
    auto xT = convFPropAttr.getX(); // NHWC
    auto wT = convFPropAttr.getW(); // KCRS
    auto yT = convFPropAttr.getY(); // NKPQ

    const auto &xDim = xT->getDim();
    const auto &wDim = wT->getDim();
    const auto &yDim = yT->getDim();

    // Shape and stride inference is future work
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
