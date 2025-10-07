// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the pointwise nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_POINTWISE_NODE_H
#define FUSILLI_NODE_POINTWISE_NODE_H

#include "fusilli/attributes/pointwise_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <string>
#include <unordered_map>

namespace fusilli {

class PointwiseNode : public NodeCRTP<PointwiseNode> {
public:
  PointwiseAttr pointwiseAttr;

  PointwiseNode(PointwiseAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), pointwiseAttr(std::move(attr)) {}

  const std::string &getName() const override final {
    return pointwiseAttr.getName();
  }
  Type getType() const override final { return Type::Pointwise; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating PointwiseNode '"
                           << pointwiseAttr.getName() << "'");
    FUSILLI_RETURN_ERROR_IF(
        pointwiseAttr.getMode() == PointwiseAttr::Mode::NOT_SET,
        ErrorCode::AttributeNotSet, "Pointwise mode not set");

    // Validate inputs based on mode
    PointwiseAttr::Mode mode = pointwiseAttr.getMode();
    int requiredCount = PointwiseAttr::modeToRequiredInputCount.at(mode);

    // Validate input requirements (required inputs must exist, unnecessary ones
    // must not)
    constexpr int maxInputs = 3;
    for (int i = 0; i < maxInputs; ++i) {
      auto inputName = static_cast<PointwiseAttr::InputNames>(i);
      bool hasInput = pointwiseAttr.inputs.contains(inputName) &&
                      pointwiseAttr.inputs.at(inputName) != nullptr;

      if (i < requiredCount) {
        FUSILLI_RETURN_ERROR_IF(!hasInput, ErrorCode::AttributeNotSet,
                                PointwiseAttr::modeToStr.at(mode) +
                                    " mode requires IN_" + std::to_string(i) +
                                    " input");
      } else {
        FUSILLI_RETURN_ERROR_IF(hasInput, ErrorCode::InvalidAttribute,
                                PointwiseAttr::modeToStr.at(mode) +
                                    " mode should not have IN_" +
                                    std::to_string(i) + " input set");
      }
    }

    // Validate output
    FUSILLI_RETURN_ERROR_IF(!pointwiseAttr.getOUT_0(),
                            ErrorCode::AttributeNotSet,
                            "Pointwise operation requires output");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for PointwiseNode '"
                           << pointwiseAttr.getName() << "'");

    // Fill missing properties from context (including data types)
    pointwiseAttr.fillFromContext(context);

    const auto &outTensor = pointwiseAttr.getOUT_0();
    if (outTensor->getDim().empty()) {
      // Collect all input shapes
      std::vector<std::vector<int64_t>> inputShapes;
      for (const auto &[_, inTensor] : pointwiseAttr.inputs)
        if (inTensor)
          inputShapes.push_back(inTensor->getDim());

      outTensor->setDim(FUSILLI_TRY(computeBroadcastShape(inputShapes)));
    }

    if (outTensor->getStride().empty()) {
      // Try to set the stride from an input shape that matches the output
      // shape.
      for (const auto &[_, inTensor] : pointwiseAttr.inputs) {
        if (!inTensor)
          continue;
        if (inTensor->getDim() != outTensor->getDim())
          continue;
        outTensor->setStride(inTensor->getStride());
      }

      if (outTensor->getStride().empty() && outTensor->isVirtual()) {
        // If we haven't found the stride already and the output is virtual,
        // compute an output stride that has the same format as IN_0. This can
        // occur when all inputs are broadcasted.
        auto inputStride = pointwiseAttr.getIN_0()->getStride();
        std::vector<size_t> strideOrder = generateStrideOrderPreservingFormat(
            inputStride, outTensor->getDim().size());
        outTensor->setStride(
            generateStrideFromDim(outTensor->getDim(), strideOrder));
      }
      FUSILLI_RETURN_ERROR_IF(outTensor->getStride().empty(),
                              ErrorCode::InvalidAttribute,
                              "Pointwise output strides could not be computed");
    }

    return ok();
  }
};
} // namespace fusilli

#endif // FUSILLI_NODE_POINTWISE_NODE_H
