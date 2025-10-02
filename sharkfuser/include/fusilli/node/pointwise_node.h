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

#include <memory>
#include <string>
#include <unordered_map>

namespace fusilli {

class PointwiseNode : public NodeCRTP<PointwiseNode> {
public:
  PointwiseAttr pointwiseAttr;

  PointwiseNode(PointwiseAttr attr, const Context &ctx)
      : NodeCRTP(ctx), pointwiseAttr(std::move(attr)) {}

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final { return ""; };
  std::string getOperandNamesAsm() const override final { return ""; };
  std::string getOperandTypesAsm() const override final { return ""; };
  std::string getResultNamesAsm() const override final { return ""; };
  std::string getResultTypesAsm() const override final { return ""; };
  std::string getResultNamesAndTypesAsm() const override final { return ""; };

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

    static const std::unordered_map<PointwiseAttr::Mode, int>
        requiredInputCount = {{PointwiseAttr::Mode::RELU, 1},
                              {PointwiseAttr::Mode::ADD, 2}};
    int requiredCount = requiredInputCount.at(mode);

    // Validate input requirements (required inputs must exist, unnecessary ones
    // must not)
    constexpr int maxInputs = 3;
    for (int i = 0; i < maxInputs; ++i) {
      auto inputName = static_cast<PointwiseAttr::InputNames>(i);
      bool hasInput = pointwiseAttr.inputs.contains(inputName) &&
                      pointwiseAttr.inputs.at(inputName) != nullptr;

      if (i < requiredCount) {
        FUSILLI_RETURN_ERROR_IF(!hasInput, ErrorCode::AttributeNotSet,
                                PointwiseAttr::modeToString(mode) +
                                    " mode requires IN" + std::to_string(i) +
                                    " input");
      } else {
        FUSILLI_RETURN_ERROR_IF(hasInput, ErrorCode::InvalidAttribute,
                                PointwiseAttr::modeToString(mode) +
                                    " mode should not have IN" +
                                    std::to_string(i) + " input set");
      }
    }

    // Validate output
    FUSILLI_RETURN_ERROR_IF(!pointwiseAttr.getOUT(), ErrorCode::AttributeNotSet,
                            "Pointwise operation requires output");

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for PointwiseNode '"
                           << pointwiseAttr.getName() << "'");

    // Fill missing properties from context (including data types)
    pointwiseAttr.fillFromContext(context);

    const auto &outTensor = pointwiseAttr.getOUT();
    if (outTensor->getDim().empty()) {
      std::vector<std::vector<int64_t>> inputShapes;
      for (const auto &[inName, inTensor] : pointwiseAttr.inputs) {
        if (inTensor) {
          inputShapes.push_back(inTensor->getDim());
        }
      }
      ErrorOr<std::vector<int64_t>> shape = computeBroadcastShapes(inputShapes);
      FUSILLI_CHECK_ERROR(shape);
      outTensor->setDim(std::move(*shape));
    }

    if (outTensor->getStride().empty()) {
      for (const auto &[inName, inTensor] : pointwiseAttr.inputs) {
        if (!inTensor) {
          continue;
        }
        if (inTensor->getDim() != outTensor->getDim()) {
          continue;
        }
        outTensor->setStride(inTensor->getStride());
      }
      FUSILLI_RETURN_ERROR_IF(outTensor->getStride().empty(),
                              ErrorCode::InvalidAttribute,
                              "Pointwise output strides could not be computed");
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating PointwiseNode '"
                           << pointwiseAttr.getName() << "'");
    return ok();
  }
};
} // namespace fusilli

#endif // FUSILLI_NODE_POINTWISE_NODE_H
