// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_GRAPH_H
#define FUSILI_GRAPH_H

#include "fusili/attributes/tensor_attributes.h"
#include "fusili/context.h"
#include "fusili/logging.h"
#include "fusili/node/conv_node.h"
#include "fusili/node/node.h"

#include <memory>
#include <unordered_set>

namespace fusili {

class Graph : public INode {
public:
  Graph() : INode(Context{}) {}

  error_t validate() {
    FUSILI_LOG_LABEL_ENDL("INFO: Validating graph");

    // Validate inputs
    for (const auto &input : fullGraphInputs_) {
      FUSILI_CHECK_ERROR(input->validate());
    }

    // Validate nodes (this infers missing tensor properties)
    FUSILI_CHECK_ERROR(validateSubtree());

    // Validate outputs
    for (const auto &output : fullGraphOutputs_) {
      FUSILI_CHECK_ERROR(output->validate());
    }

    // Check for uid uniqueness (when pre-assigned)
    FUSILI_CHECK_ERROR(checkPreAssignedUidsAreUnique())

    return {error_code_t::OK, ""};
  }

  Type getType() override { return Type::Composite; }

  Graph &setIODataType(DataType type) {
    context.setIODataType(type);
    return *this;
  }

  Graph &setComputeDataType(DataType type) {
    context.setComputeDataType(type);
    return *this;
  }

  Graph &setIntermediateDataType(DataType type) {
    context.setIntermediateDataType(type);
    return *this;
  }

  error_t queryTensorOfUid(int64_t uid, TensorAttr &tensor) const {
    for (const auto &iTensor : fullGraphInputs_) {
      if (iTensor->getUid() == uid) {
        tensor = *iTensor;
        return {error_code_t::OK, ""};
      }
    }
    for (const auto &oTensor : fullGraphOutputs_) {
      if (oTensor->getUid() == uid) {
        tensor = *oTensor;
        return {error_code_t::OK, ""};
      }
    }
    return {error_code_t::TensorNotFound,
            "Tensor with UID " + std::to_string(uid) + " not found"};
  }

  std::shared_ptr<TensorAttr> tensor(const TensorAttr &tensor);

  std::shared_ptr<TensorAttr> convFProp(const std::shared_ptr<TensorAttr> &x,
                                        const std::shared_ptr<TensorAttr> &w,
                                        ConvFPropAttr &attributes);

private:
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphInputs_;
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphOutputs_;
  std::unordered_set<TensorAttr::uid_t> usedUids_;

  std::shared_ptr<TensorAttr> outputTensor(const std::string &name) {
    auto tensor = std::make_shared<TensorAttr>();
    tensor->setName(name).setIsVirtual(true);
    fullGraphOutputs_.insert(tensor);
    return tensor;
  }

  error_t preValidateNode() const override final {
    return {error_code_t::OK, ""};
  }

  error_t inferPropertiesNode() override final {
    return {error_code_t::OK, ""};
  }

  error_t postValidateNode() const override final {
    return {error_code_t::OK, ""};
  }

  error_t checkPreAssignedUidsAreUnique() {
    usedUids_.clear();

    for (const auto &input : fullGraphInputs_) {
      if (input->hasUid()) {
        auto uid = input->getUid();
        FUSILI_RETURN_ERROR_IF(usedUids_.find(uid) != usedUids_.end(),
                               error_code_t::InvalidAttribute,
                               "Tensor named " + input->getName() +
                                   " uses UID " + std::to_string(uid) +
                                   " which has already been assigned to "
                                   "another tensor in the graph");
        usedUids_.insert(uid);
      }
    }

    for (const auto &output : fullGraphOutputs_) {
      if (output->hasUid()) {
        auto uid = output->getUid();
        FUSILI_RETURN_ERROR_IF(usedUids_.find(uid) != usedUids_.end(),
                               error_code_t::InvalidAttribute,
                               "Tensor named " + output->getName() +
                                   " uses UID " + std::to_string(uid) +
                                   " which has already been assigned to "
                                   "another tensor in the graph");
        usedUids_.insert(uid);
      }
    }

    return {error_code_t::OK, ""};
  }
};

// Given a TensorAttr, create a shared pointer and add it to the graph's
// inputs. This allows the graph to manage the lifetime of the input tensor.
inline std::shared_ptr<TensorAttr> Graph::tensor(const TensorAttr &tensor) {
  auto tensorPtr = std::make_shared<TensorAttr>(tensor);
  fullGraphInputs_.insert(tensorPtr);
  return tensorPtr;
}

inline std::shared_ptr<TensorAttr>
Graph::convFProp(const std::shared_ptr<TensorAttr> &x,
                 const std::shared_ptr<TensorAttr> &w,
                 ConvFPropAttr &convAttr) {
  // Populate names when not set
  if (convAttr.getName().empty())
    convAttr.setName("conv_fprop_" + std::to_string(subNodes_.size()));
  if (x->getName().empty())
    x->setName(convAttr.getName() + "::X");
  if (w->getName().empty())
    w->setName(convAttr.getName() + "::W");

  // Set inputs
  convAttr.setX(x).setW(w);

  // Set outputs
  auto y = outputTensor(convAttr.getName() + "::Y");
  convAttr.setY(y);

  // Create node and add to subNodes_
  subNodes_.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(convAttr), context));

  return y;
}

} // namespace fusili

#endif // FUSILI_GRAPH_H
