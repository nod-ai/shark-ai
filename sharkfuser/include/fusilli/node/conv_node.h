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
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <string>

namespace fusilli {

class ConvFPropNode : public NodeCRTP<ConvFPropNode> {
public:
  ConvFPropAttr convFPropAttr;

  ConvFPropNode(ConvFPropAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), convFPropAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const override final;
  std::string getOperandTypesAsm() const override final;
  std::string getResultNamesAsm() const override final;
  std::string getResultTypesAsm() const override final;
  std::string getStrideOpsAsm() const;
  std::string getPaddingOpsAsm() const;
  std::string getDilationOpsAsm() const;

  const std::string &getName() const override final {
    return convFPropAttr.getName();
  }
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

    // Logical layout is always channels-first (NCHW if 4D)
    auto xT = convFPropAttr.getX(); // NCHW if 4D
    auto wT = convFPropAttr.getW(); // KCRS if 4D
    auto yT = convFPropAttr.getY(); // NKPQ if 4D

    const auto &xDim = xT->getDim();
    const auto &wDim = wT->getDim();

    auto yDim = yT->getDim();
    auto yStride = yT->getStride();

    // Infer shape of output tensor
    if (yDim.empty()) {
      yDim.resize(xDim.size());
      // N
      yDim[0] = xDim[0];
      // K
      yDim[1] = wDim[0];
      // PQ...
      for (size_t i = 2; i < xDim.size(); ++i) {
        yDim[i] = (xDim[i] + 2 * convFPropAttr.getPadding()[i - 2] -
                   (wDim[i] - 1) * convFPropAttr.getDilation()[i - 2] - 1) /
                      convFPropAttr.getStride()[i - 2] +
                  1;
      }
      yT->setDim(yDim);
    }

    // Infer stride of output tensor
    if (yStride.empty()) {
      // When unspecified, preserve the stride order of xT (input tensor)
      yStride = generateStrideFromDim(
          yDim, getStrideOrderFromStride(xT->getStride()));
      yT->setStride(yStride);
    }

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_CONV_NODE_H
