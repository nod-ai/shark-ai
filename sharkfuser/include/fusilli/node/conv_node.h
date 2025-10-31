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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
// Helper functions for convolution nodes.
//===----------------------------------------------------------------------===//

// Infer the output shape of a convolution operation from the input and weight
// shapes, dilation, padding, and stride.
inline std::vector<int64_t> getConvInferredOutputShape(
    const std::vector<int64_t> &xDim, const std::vector<int64_t> &wDim,
    const std::vector<int64_t> &dilation, const std::vector<int64_t> &padding,
    const std::vector<int64_t> &stride) {
  constexpr size_t kSpatialStartIdx = 2;
  std::vector<int64_t> yDim(xDim.size());

  // N (batch dim)
  yDim[0] = xDim[0];
  // K (channel dim)
  yDim[1] = wDim[0];
  // PQ... (spatial dims)
  for (size_t i = kSpatialStartIdx; i < xDim.size(); ++i) {
    yDim[i] = 1 + (xDim[i] - (wDim[i] - 1) * dilation[i - kSpatialStartIdx] +
                   2 * padding[i - kSpatialStartIdx] - 1) /
                      stride[i - kSpatialStartIdx];
  }
  return yDim;
}

//===----------------------------------------------------------------------===//
// Convolution nodes.
//===----------------------------------------------------------------------===//

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
  std::string getGroupOpsAsm() const;
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

    const std::vector<int64_t> &padding = convFPropAttr.getPadding();
    const std::vector<int64_t> &stride = convFPropAttr.getStride();
    const std::vector<int64_t> &dilation = convFPropAttr.getDilation();

    FUSILLI_RETURN_ERROR_IF(padding.empty(), ErrorCode::AttributeNotSet,
                            "Conv padding not set");
    FUSILLI_RETURN_ERROR_IF(stride.empty(), ErrorCode::AttributeNotSet,
                            "Conv stride not set");
    FUSILLI_RETURN_ERROR_IF(dilation.empty(), ErrorCode::AttributeNotSet,
                            "Conv dilation not set");

    std::shared_ptr<TensorAttr> xT = convFPropAttr.getX();
    std::shared_ptr<TensorAttr> wT = convFPropAttr.getW();
    std::shared_ptr<TensorAttr> yT = convFPropAttr.getY();

    // Ensure input and weight tensors are set.
    FUSILLI_RETURN_ERROR_IF(!xT, ErrorCode::AttributeNotSet,
                            "Conv input tensor X not set");
    FUSILLI_RETURN_ERROR_IF(!wT, ErrorCode::AttributeNotSet,
                            "Conv weight tensor W not set");
    FUSILLI_RETURN_ERROR_IF(!yT, ErrorCode::AttributeNotSet,
                            "Conv output tensor Y not set");

    size_t xRank = xT->getDim().size();
    size_t wRank = wT->getDim().size();

    // Rank checks on input and weight tensors.
    FUSILLI_RETURN_ERROR_IF(
        xRank < 3, ErrorCode::InvalidAttribute,
        "Conv input tensor X must have a rank of at least 3");
    FUSILLI_RETURN_ERROR_IF(
        xRank != wRank, ErrorCode::InvalidAttribute,
        "Conv input tensor X and weight tensor W have different ranks");

    // Check padding, stride and dilation match rank of conv
    // All dims except batch and channel (feature) are spatial dims
    size_t numSpatialDims = xRank - 2;
    FUSILLI_RETURN_ERROR_IF(
        padding.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "Conv padding size does not match number of spatial dimensions");
    FUSILLI_RETURN_ERROR_IF(
        stride.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "Conv stride size does not match number of spatial dimensions");
    FUSILLI_RETURN_ERROR_IF(
        dilation.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "Conv dilation size does not match number of spatial dimensions");

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

    // Group count checks
    constexpr size_t inChannelsIdx = 1;
    constexpr size_t outChannelsIdx = 0;
    int64_t inChannels = xT->getDim()[inChannelsIdx];
    int64_t outChannels = wT->getDim()[outChannelsIdx];
    int64_t filterChannels = wT->getDim()[inChannelsIdx];
    FUSILLI_RETURN_ERROR_IF(
        inChannels % filterChannels != 0, ErrorCode::InvalidAttribute,
        "Conv input channels must be divisible by the filter channels");

    int64_t groupCount = inChannels / filterChannels;
    FUSILLI_RETURN_ERROR_IF(
        groupCount <= 0 || groupCount > inChannels || groupCount > outChannels,
        ErrorCode::InvalidAttribute,
        "Conv group count must be greater than 0 and less than or equal to the "
        "numbers of input and outputs channels");
    FUSILLI_RETURN_ERROR_IF(
        outChannels % groupCount != 0, ErrorCode::InvalidAttribute,
        "Conv output channels must be divisible by the group count");

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

    // Infer shape of output tensor.
    if (yDim.empty()) {
      yDim = getConvInferredOutputShape(xDim, wDim, dilation, padding, stride);
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

    std::shared_ptr<TensorAttr> xT = convFPropAttr.getX();
    std::shared_ptr<TensorAttr> wT = convFPropAttr.getW();
    std::shared_ptr<TensorAttr> yT = convFPropAttr.getY();

    size_t xRank = xT->getDim().size();
    size_t yRank = yT->getDim().size();

    // Rank checks
    FUSILLI_RETURN_ERROR_IF(
        yRank < 3, ErrorCode::InvalidAttribute,
        "Conv output tensor Y must have a rank of at least 3");
    FUSILLI_RETURN_ERROR_IF(
        xRank != yRank, ErrorCode::InvalidAttribute,
        "Conv input tensor X and output tensor Y have different ranks");

    FUSILLI_RETURN_ERROR_IF(
        yT->getDim() != getConvInferredOutputShape(xT->getDim(), wT->getDim(),
                                                   convFPropAttr.getDilation(),
                                                   convFPropAttr.getPadding(),
                                                   convFPropAttr.getStride()),
        ErrorCode::InvalidAttribute,
        "Conv output tensor Y dimensions do not match the expected shapes "
        "inferred based on the input and weight dimensions");

    // Contiguity check for output tensor.
    // When output strides are not specified, they are inferred and will be
    // correct by construction. This check is for when output strides are
    // specified by the user.
    FUSILLI_RETURN_ERROR_IF(!yT->isContiguous() && !yT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + yT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    return ok();
  }
};

class ConvWGradNode : public NodeCRTP<ConvWGradNode> {
public:
  ConvWGradAttr convWGradAttr;

  ConvWGradNode(ConvWGradAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), convWGradAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;
  std::string getStrideOpsAsm() const;
  std::string getPaddingOpsAsm() const;
  std::string getDilationOpsAsm() const;
  std::string getPermuteDYOpsAsm() const;
  std::string getPermuteXOpsAsm() const;
  std::string getPermuteDWOpsAsm() const;
  std::string getPermuteEmptyWOpsAsm() const;

  const std::string &getName() const override final {
    return convWGradAttr.getName();
  }
  Type getType() const override final { return Type::WGrad; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating ConvWGradNode '"
                           << convWGradAttr.getName() << "'");

    const std::vector<int64_t> &padding = convWGradAttr.getPadding();
    const std::vector<int64_t> &stride = convWGradAttr.getStride();
    const std::vector<int64_t> &dilation = convWGradAttr.getDilation();
    FUSILLI_RETURN_ERROR_IF(padding.empty(), ErrorCode::AttributeNotSet,
                            "ConvWGrad padding not set");
    FUSILLI_RETURN_ERROR_IF(stride.empty(), ErrorCode::AttributeNotSet,
                            "ConvWGrad stride not set");
    FUSILLI_RETURN_ERROR_IF(dilation.empty(), ErrorCode::AttributeNotSet,
                            "ConvWGrad dilation not set");

    std::shared_ptr<TensorAttr> dyT = convWGradAttr.getDY();
    std::shared_ptr<TensorAttr> xT = convWGradAttr.getX();
    std::shared_ptr<TensorAttr> dwT = convWGradAttr.getDW();

    // Ensure input and weight tensors are set.
    FUSILLI_RETURN_ERROR_IF(!dyT, ErrorCode::AttributeNotSet,
                            "ConvWGrad gradient tensor DY not set");
    FUSILLI_RETURN_ERROR_IF(!xT, ErrorCode::AttributeNotSet,
                            "ConvWGrad input tensor X not set");
    FUSILLI_RETURN_ERROR_IF(!dwT, ErrorCode::AttributeNotSet,
                            "ConvWGrad output tensor DW not set");

    // Rank checks on DY and X tensors.
    size_t dyRank = dyT->getDim().size();
    size_t xRank = xT->getDim().size();

    FUSILLI_RETURN_ERROR_IF(
        dyRank < 3 || xRank < 3, ErrorCode::InvalidAttribute,
        "ConvWGrad input tensors DY/X must have a rank of at least 3");
    FUSILLI_RETURN_ERROR_IF(dyRank != xRank, ErrorCode::InvalidAttribute,
                            "ConvWGrad tensors DY and X have different ranks");

    // Check padding, stride and dilation match rank of conv
    // All dims except batch and channel (feature) are spatial dims
    size_t numSpatialDims = dyRank - 2;
    FUSILLI_RETURN_ERROR_IF(
        padding.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "ConvWGrad padding size does not match number of spatial dimensions");
    FUSILLI_RETURN_ERROR_IF(
        stride.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "ConvWGrad stride size does not match number of spatial dimensions");
    FUSILLI_RETURN_ERROR_IF(
        dilation.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "ConvWGrad dilation size does not match number of spatial dimensions");

    // Layout checks on input tensors.
    FUSILLI_RETURN_ERROR_IF(!dyT->isContiguous() && !dyT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + dyT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");
    FUSILLI_RETURN_ERROR_IF(!xT->isContiguous() && !xT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + xT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");
    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for ConvWGradNode '"
                           << convWGradAttr.getName() << "'");

    convWGradAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> dyT = convWGradAttr.getDY();
    std::shared_ptr<TensorAttr> dwT = convWGradAttr.getDW();
    const std::vector<int64_t> &wDim = dwT->getDim();

    // Can't infer the output dims because we don't know the number of groups
    // and `output_channels = input_channels / groups`. Only infer stride of
    // weight tensor.
    std::vector<int64_t> wStride = dwT->getStride();
    if (wStride.empty()) {
      // When unspecified, preserve the stride order of dyT (gradient tensor).
      wStride = dyT->isContiguous()
                    ? generateStrideFromDim(
                          wDim, getContiguousStrideOrder(wDim.size()))
                    : generateStrideFromDim(
                          wDim, getChannelsLastStrideOrder(wDim.size()));

      dwT->setStride(std::move(wStride));
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating ConvWGradNode '"
                           << convWGradAttr.getName() << "'");

    std::shared_ptr<TensorAttr> dyT = convWGradAttr.getDY();
    std::shared_ptr<TensorAttr> xT = convWGradAttr.getX();
    std::shared_ptr<TensorAttr> dwT = convWGradAttr.getDW();

    size_t dwRank = dwT->getDim().size();
    FUSILLI_RETURN_ERROR_IF(
        dwRank < 3, ErrorCode::InvalidAttribute,
        "ConvWGrad weight gradient tensor DW must have a rank of at least 3");

    FUSILLI_RETURN_ERROR_IF(
        dyT->getDim() != getConvInferredOutputShape(xT->getDim(), dwT->getDim(),
                                                    convWGradAttr.getDilation(),
                                                    convWGradAttr.getPadding(),
                                                    convWGradAttr.getStride()),
        ErrorCode::InvalidAttribute,
        "ConvWGrad output DW dimensions do not match the expected shapes "
        "inferred based on input dimensions");

    // Contiguity check for output tensor.
    // When output strides are not specified, they are inferred and will be
    // correct by construction. This check is for when output strides are
    // specified by the user.
    FUSILLI_RETURN_ERROR_IF(!dwT->isContiguous() && !dwT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + dwT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    return ok();
  }
};

class ConvDGradNode : public NodeCRTP<ConvDGradNode> {
public:
  ConvDGradAttr convDGradAttr;

  ConvDGradNode(ConvDGradAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), convDGradAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string getOperandNamesAsm() const;
  std::string getOperandTypesAsm() const;
  std::string getResultNamesAsm() const;
  std::string getResultTypesAsm() const;
  std::string getGroupOpsAsm() const;
  std::string getStrideOpsAsm() const;
  std::string getPaddingOpsAsm() const;
  std::string getDilationOpsAsm() const;
  std::string getPermuteDYOpsAsm() const;
  std::string getPermuteWOpsAsm() const;
  std::string getPermuteDXOpsAsm() const;
  std::string getPermuteEmptyXOpsAsm() const;

  const std::string &getName() const override final {
    return convDGradAttr.getName();
  }
  Type getType() const override final { return Type::DGrad; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating ConvDGradNode '"
                           << convDGradAttr.getName() << "'");

    const std::vector<int64_t> &padding = convDGradAttr.getPadding();
    const std::vector<int64_t> &stride = convDGradAttr.getStride();
    const std::vector<int64_t> &dilation = convDGradAttr.getDilation();
    FUSILLI_RETURN_ERROR_IF(padding.empty(), ErrorCode::AttributeNotSet,
                            "ConvDGrad padding not set");
    FUSILLI_RETURN_ERROR_IF(stride.empty(), ErrorCode::AttributeNotSet,
                            "ConvDGrad stride not set");
    FUSILLI_RETURN_ERROR_IF(dilation.empty(), ErrorCode::AttributeNotSet,
                            "ConvDGrad dilation not set");

    std::shared_ptr<TensorAttr> dyT = convDGradAttr.getDY();
    std::shared_ptr<TensorAttr> wT = convDGradAttr.getW();
    std::shared_ptr<TensorAttr> dxT = convDGradAttr.getDX();

    // Ensure input and weight tensors are set.
    FUSILLI_RETURN_ERROR_IF(!dyT, ErrorCode::AttributeNotSet,
                            "ConvDGrad gradient tensor DY not set");
    FUSILLI_RETURN_ERROR_IF(!wT, ErrorCode::AttributeNotSet,
                            "ConvDGrad input tensor W not set");
    FUSILLI_RETURN_ERROR_IF(!dxT, ErrorCode::AttributeNotSet,
                            "ConvDGrad output tensor DX not set");

    size_t dyRank = dyT->getDim().size();
    size_t wRank = wT->getDim().size();

    FUSILLI_RETURN_ERROR_IF(
        dyRank < 3 || wRank < 3, ErrorCode::InvalidAttribute,
        "ConvDGrad input tensors DY/W must have a rank of at least 3");

    // Check padding, stride and dilation match rank of conv
    // All dims except batch and channel (feature) are spatial dims
    size_t numSpatialDims = dyRank - 2;
    FUSILLI_RETURN_ERROR_IF(
        padding.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "ConvDGrad padding size does not match number of spatial dimensions");
    FUSILLI_RETURN_ERROR_IF(
        stride.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "ConvDGrad stride size does not match number of spatial dimensions");
    FUSILLI_RETURN_ERROR_IF(
        dilation.size() != numSpatialDims, ErrorCode::InvalidAttribute,
        "ConvDGrad dilation size does not match number of spatial dimensions");

    // Layout checks on input tensors.
    FUSILLI_RETURN_ERROR_IF(!dyT->isContiguous() && !dyT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + dyT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");
    FUSILLI_RETURN_ERROR_IF(!wT->isContiguous() && !wT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + wT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    // Group count checks
    constexpr size_t inChannelsIdx = 1;
    constexpr size_t outChannelsIdx = 0;
    int64_t inChannels = dxT->getDim()[inChannelsIdx];
    int64_t outChannels = wT->getDim()[outChannelsIdx];
    int64_t filterChannels = wT->getDim()[inChannelsIdx];
    FUSILLI_RETURN_ERROR_IF(inChannels % filterChannels != 0,
                            ErrorCode::InvalidAttribute,
                            "ConvDGrad input (DX) channels must be divisible "
                            "by the filter (W) channels");

    int64_t groupCount = inChannels / filterChannels;
    FUSILLI_RETURN_ERROR_IF(groupCount <= 0 || groupCount > inChannels ||
                                groupCount > outChannels,
                            ErrorCode::InvalidAttribute,
                            "ConvDGrad group count must be greater than 0 and "
                            "less than or equal to the "
                            "numbers of input and outputs channels");
    FUSILLI_RETURN_ERROR_IF(
        outChannels % groupCount != 0, ErrorCode::InvalidAttribute,
        "ConvDGrad output (DY) channels must be divisible by the group count");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for ConvDGradNode '"
                           << convDGradAttr.getName() << "'");

    convDGradAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> dyT = convDGradAttr.getDY();
    std::shared_ptr<TensorAttr> dxT = convDGradAttr.getDX();
    const std::vector<int64_t> &dxDim = dxT->getDim();
    const std::vector<int64_t> &dxStride = dxT->getStride();

    // Can't infer the output dims because we don't know the number of groups
    // and `input_channels = output_channels * groups`.
    // TODO: hipdnn allows inferring the dims by assuming that the group count
    // is one but cudnn doesn't allow this to be inferred. Only infer stride of
    // the DX tensor.
    if (dxStride.empty()) {
      // When unspecified, preserve the stride order of dyT.
      dxT->setStride(
          dyT->isContiguous()
              ? generateStrideFromDim(dxDim,
                                      getContiguousStrideOrder(dxDim.size()))
              : generateStrideFromDim(
                    dxDim, getChannelsLastStrideOrder(dxDim.size())));
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating ConvDGradNode '"
                           << convDGradAttr.getName() << "'");

    std::shared_ptr<TensorAttr> dyT = convDGradAttr.getDY();
    std::shared_ptr<TensorAttr> wT = convDGradAttr.getW();
    std::shared_ptr<TensorAttr> dxT = convDGradAttr.getDX();
    size_t dyRank = dyT->getDim().size();
    size_t dxRank = dxT->getDim().size();

    FUSILLI_RETURN_ERROR_IF(
        dxRank < 3, ErrorCode::InvalidAttribute,
        "ConvDGrad input gradient DX must have a rank of at least 3");
    FUSILLI_RETURN_ERROR_IF(dyRank != dxRank, ErrorCode::InvalidAttribute,
                            "ConvDGrad tensors DY and DX have different ranks");

    FUSILLI_RETURN_ERROR_IF(
        dyT->getDim() != getConvInferredOutputShape(dxT->getDim(), wT->getDim(),
                                                    convDGradAttr.getDilation(),
                                                    convDGradAttr.getPadding(),
                                                    convDGradAttr.getStride()),
        ErrorCode::InvalidAttribute,
        "ConvDGrad DY dimensions do not match the expected shapes inferred "
        "based on DX and W");

    // Contiguity check for output tensor.
    FUSILLI_RETURN_ERROR_IF(!dxT->isContiguous() && !dxT->isChannelsLast(),
                            ErrorCode::NotImplemented,
                            "Tensor '" + dxT->getName() +
                                "' is neither contiguous nor channels-last as "
                                "defined by its stride");

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_CONV_NODE_H
