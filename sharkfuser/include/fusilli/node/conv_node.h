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

#include <memory>
#include <string>

namespace fusilli {

class ConvFPropNode : public NodeCRTP<ConvFPropNode> {
public:
  ConvFPropAttr convFPropAttr;

  ConvFPropNode(ConvFPropAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), convFPropAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;
  std::string getStrideOpsAsm() const;
  std::string getPaddingOpsAsm() const;
  std::string getDilationOpsAsm() const;
  std::string getPermuteXOpsAsm() const;
  std::string getPermuteWOpsAsm() const;
  std::string getPermuteYOpsAsm() const;

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

    std::shared_ptr<TensorAttr> xT = convFPropAttr.getX();
    std::shared_ptr<TensorAttr> wT = convFPropAttr.getW();

    // Ensure input and weight tensors are set.
    FUSILLI_RETURN_ERROR_IF(!xT, ErrorCode::AttributeNotSet,
                            "Conv input tensor X not set");
    FUSILLI_RETURN_ERROR_IF(!wT, ErrorCode::AttributeNotSet,
                            "Conv weight tensor W not set");

    // Layout checks on input and weight tensors.
    FUSILLI_RETURN_ERROR_IF(!xT->isContiguous() && !xT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + xT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");
    FUSILLI_RETURN_ERROR_IF(!wT->isContiguous() && !wT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + wT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for ConvFPropNode '"
                           << convFPropAttr.getName() << "'");

    convFPropAttr.fillFromContext(context);

    // Logical layout is always channels-first (NCHW if 4D).
    std::shared_ptr<TensorAttr> xT = convFPropAttr.getX(); // NCHW if 4D
    std::shared_ptr<TensorAttr> wT = convFPropAttr.getW(); // KCRS if 4D
    std::shared_ptr<TensorAttr> yT = convFPropAttr.getY(); // NKPQ if 4D

    const std::vector<int64_t> &dilation = convFPropAttr.getDilation();
    const std::vector<int64_t> &padding = convFPropAttr.getPadding();
    const std::vector<int64_t> &stride = convFPropAttr.getStride();

    const std::vector<int64_t> &xDim = xT->getDim();
    const std::vector<int64_t> &wDim = wT->getDim();

    std::vector<int64_t> yDim = yT->getDim();
    std::vector<int64_t> yStride = yT->getStride();

    // For spatial layouts (3D and above), we expect the spatial dims
    // to start at index = 2 (after batch and channel dims).
    constexpr size_t kSpatialStartIdx = 2;

    // Infer shape of output tensor.
    if (yDim.empty()) {
      yDim.resize(xDim.size());
      // N (batch dim)
      yDim[0] = xDim[0];
      // K (channel dim)
      yDim[1] = wDim[0];
      // PQ... (spatial dims)
      for (size_t i = kSpatialStartIdx; i < xDim.size(); ++i) {
        yDim[i] =
            1 + (xDim[i] - (wDim[i] - 1) * dilation[i - kSpatialStartIdx] +
                 2 * padding[i - kSpatialStartIdx] - 1) /
                    stride[i - kSpatialStartIdx];
      }
      yT->setDim(yDim);
    }

    // Infer stride of output tensor.
    if (yStride.empty()) {
      // When unspecified, preserve the stride order of xT (input tensor).
      yStride = xT->isContiguous()
                    ? generateStrideFromDim(
                          yDim, getContiguousStrideOrder(yDim.size()))
                    : generateStrideFromDim(
                          yDim, getChannelsLastStrideOrder(yDim.size()));

      yT->setStride(yStride);
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating ConvFPropNode '"
                           << convFPropAttr.getName() << "'");

    // Contiguity check for output tensor.
    // When output strides are not specified, they are inferred and will be
    // correct by construction. This check is for when output strides are
    // specified by the user.
    std::shared_ptr<TensorAttr> yT = convFPropAttr.getY();
    FUSILLI_RETURN_ERROR_IF(!yT->isContiguous() && !yT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + yT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_CONV_NODE_H
