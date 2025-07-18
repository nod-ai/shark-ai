// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_GRAPH_H
#define FUSILI_GRAPH_H

#include <memory>
#include <unordered_set>

#include "fusili/attributes/tensor_attributes.h"
#include "fusili/context.h"
#include "fusili/logging.h"
#include "fusili/node/conv_node.h"
#include "fusili/node/node.h"

namespace fusili {

class Graph : public INode {
private:
  std::unordered_set<std::shared_ptr<TensorAttr>> full_graph_inputs;
  std::unordered_set<std::shared_ptr<TensorAttr>> full_graph_outputs;
  std::unordered_set<TensorAttr::uid_t> used_uids;

  std::shared_ptr<TensorAttr> output_tensor(std::string const &name) {
    auto tensor = std::make_shared<TensorAttr>();
    tensor->setName(name).setIsVirtual(true);
    full_graph_outputs.insert(tensor);
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

  error_t check_pre_assigned_uids_are_unique() {
    used_uids.clear();

    for (auto const &input : full_graph_inputs) {
      if (input->hasUid()) {
        auto uid = input->getUid();
        FUSILI_RETURN_ERROR_IF(used_uids.find(uid) != used_uids.end(),
                               error_code_t::INVALID_ATTRIBUTE,
                               "Tensor named " + input->getName() +
                                   " uses UID " + std::to_string(uid) +
                                   " which has already been assigned to "
                                   "another tensor in the graph");
        used_uids.insert(uid);
      }
    }

    for (auto const &output : full_graph_outputs) {
      if (output->hasUid()) {
        auto uid = output->getUid();
        FUSILI_RETURN_ERROR_IF(used_uids.find(uid) != used_uids.end(),
                               error_code_t::INVALID_ATTRIBUTE,
                               "Tensor named " + output->getName() +
                                   " uses UID " + std::to_string(uid) +
                                   " which has already been assigned to "
                                   "another tensor in the graph");
        used_uids.insert(uid);
      }
    }

    return {error_code_t::OK, ""};
  }

public:
  Graph() : INode(Context{}) {}

  error_t validate() {
    FUSILI_LOG_LABEL_ENDL("INFO: Validating graph");

    // Validate inputs
    for (auto const &input : full_graph_inputs) {
      FUSILI_CHECK_ERROR(input->validate());
    }

    // Validate nodes (this infers missing tensor properties)
    FUSILI_CHECK_ERROR(validateSubtree());

    // Validate outputs
    for (auto const &output : full_graph_outputs) {
      FUSILI_CHECK_ERROR(output->validate());
    }

    // Check for uid uniqueness (when pre-assigned)
    FUSILI_CHECK_ERROR(check_pre_assigned_uids_are_unique())

    return {error_code_t::OK, ""};
  }

  Type getType() override { return Type::Composite; }

  Graph &set_io_data_type(DataType_t const type) {
    context.setIODataType(type);
    return *this;
  }

  Graph &set_compute_data_type(DataType_t const type) {
    context.setComputeDataType(type);
    return *this;
  }

  Graph &set_intermediate_data_type(DataType_t const type) {
    context.setIntermediateDataType(type);
    return *this;
  }

  error_t query_tensor_of_uid(int64_t const uid, TensorAttr &tensor) const {
    for (auto const &i_tensor : full_graph_inputs) {
      if (i_tensor->getUid() == uid) {
        tensor = *i_tensor;
        return {error_code_t::OK, ""};
      }
    }
    for (auto const &o_tensor : full_graph_outputs) {
      if (o_tensor->getUid() == uid) {
        tensor = *o_tensor;
        return {error_code_t::OK, ""};
      }
    }
    return {error_code_t::TENSOR_NOT_FOUND,
            "Tensor with UID " + std::to_string(uid) + " not found"};
  }

  std::shared_ptr<TensorAttr> tensor(TensorAttr const &tensor);

  std::shared_ptr<TensorAttr> conv_fprop(std::shared_ptr<TensorAttr> const &x,
                                         std::shared_ptr<TensorAttr> const &w,
                                         ConvFPropAttr &attributes);
};

// Given a TensorAttr, create a shared pointer and add it to the graph's
// inputs. This allows the graph to manage the lifetime of the input tensor.
inline std::shared_ptr<TensorAttr> Graph::tensor(TensorAttr const &tensor) {
  auto tensor_ptr = std::make_shared<TensorAttr>(tensor);
  full_graph_inputs.insert(tensor_ptr);
  return tensor_ptr;
}

inline std::shared_ptr<TensorAttr>
Graph::conv_fprop(std::shared_ptr<TensorAttr> const &x,
                  std::shared_ptr<TensorAttr> const &w,
                  ConvFPropAttr &conv_attr) {
  // Populate names when not set
  if (conv_attr.getName().empty())
    conv_attr.setName("conv_fprop_" + std::to_string(subNodes_.size()));
  if (x->getName().empty())
    x->setName(conv_attr.getName() + "::X");
  if (w->getName().empty())
    w->setName(conv_attr.getName() + "::W");

  // Set inputs
  conv_attr.setX(x).setW(w);

  // Set outputs
  auto y = output_tensor(conv_attr.getName() + "::Y");
  conv_attr.setY(y);

  // Create node and add to subNodes_
  subNodes_.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(conv_attr), context));

  return y;
}

} // namespace fusili

#endif // FUSILI_GRAPH_H
