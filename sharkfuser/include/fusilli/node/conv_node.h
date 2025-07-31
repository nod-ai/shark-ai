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

#ifndef FUSILLI_NODE_CONV_NODE_H
#define FUSILLI_NODE_CONV_NODE_H

#include "fusilli/attributes/conv_attributes.h"
#include "fusilli/context.h"
#include "fusilli/logging.h"
#include "fusilli/node/node.h"

#include <string>

namespace fusilli {

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

  std::string getName() const override final { return convFPropAttr.getName(); }
  Type getType() const override final { return Type::Convolution; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating ConvFPropNode '"
                           << convFPropAttr.getName() << "'");
    FUSILLI_RETURN_ERROR_IF(convFPropAttr.getPadding().empty(),
                            ErrorCode::AttributeNotSet, "Conv padding not set");
    FUSILLI_RETURN_ERROR_IF(convFPropAttr.getStride().empty(),
                            ErrorCode::AttributeNotSet, "Conv stride not set");
    FUSILLI_RETURN_ERROR_IF(convFPropAttr.getDilation().empty(),
                            ErrorCode::AttributeNotSet,
                            "Conv dilation not set");
    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for ConvFPropNode '"
                           << convFPropAttr.getName() << "'");

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
      FUSILLI_RETURN_ERROR_IF(true, ErrorCode::NotImplemented,
                              "ConvFProp node shape inference not implemented "
                              "yet; please specify output tensor dimensions");
    }
    if (yT->getStride().empty()) {
      FUSILLI_RETURN_ERROR_IF(
          true, ErrorCode::NotImplemented,
          "ConvFProp node stride inference not implemented yet; please "
          "specify output tensor stride");
    }

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_CONV_NODE_H
