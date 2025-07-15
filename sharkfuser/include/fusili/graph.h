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
    tensor->set_name(name).set_is_virtual(true);
    full_graph_outputs.insert(tensor);
    return tensor;
  }

  error_t pre_validate_node() const override final {
    return {error_code_t::OK, ""};
  }

  error_t infer_properties_node() override final {
    return {error_code_t::OK, ""};
  }

  error_t post_validate_node() const override final {
    return {error_code_t::OK, ""};
  }

  error_t check_pre_assigned_uids_are_unique() {
    used_uids.clear();

    for (auto const &input : full_graph_inputs) {
      if (input->has_uid()) {
        auto uid = input->get_uid();
        FUSILI_RETURN_ERROR_IF(used_uids.find(uid) != used_uids.end(),
                               error_code_t::INVALID_ATTRIBUTE,
                               "Tensor named " + input->get_name() +
                                   " uses UID " + std::to_string(uid) +
                                   " which has already been assigned to "
                                   "another tensor in the graph");
        used_uids.insert(uid);
      }
    }

    for (auto const &output : full_graph_outputs) {
      if (output->has_uid()) {
        auto uid = output->get_uid();
        FUSILI_RETURN_ERROR_IF(used_uids.find(uid) != used_uids.end(),
                               error_code_t::INVALID_ATTRIBUTE,
                               "Tensor named " + output->get_name() +
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
    FUSILI_CHECK_ERROR(validate_subtree());

    // Validate outputs
    for (auto const &output : full_graph_outputs) {
      FUSILI_CHECK_ERROR(output->validate());
    }

    // Check for uid uniqueness (when pre-assigned)
    FUSILI_CHECK_ERROR(check_pre_assigned_uids_are_unique())

    return {error_code_t::OK, ""};
  }

  Type getType() override { return Type::COMPOSITE; }

  Graph &set_io_data_type(DataType_t const type) {
    context.set_io_data_type(type);
    return *this;
  }

  Graph &set_compute_data_type(DataType_t const type) {
    context.set_compute_data_type(type);
    return *this;
  }

  Graph &set_intermediate_data_type(DataType_t const type) {
    context.set_intermediate_data_type(type);
    return *this;
  }

  error_t query_tensor_of_uid(int64_t const uid, TensorAttr &tensor) const {
    for (auto const &i_tensor : full_graph_inputs) {
      if (i_tensor->get_uid() == uid) {
        tensor = *i_tensor;
        return {error_code_t::OK, ""};
      }
    }
    for (auto const &o_tensor : full_graph_outputs) {
      if (o_tensor->get_uid() == uid) {
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
  if (conv_attr.get_name().empty())
    conv_attr.set_name("conv_fprop_" + std::to_string(sub_nodes.size()));
  if (x->get_name().empty())
    x->set_name(conv_attr.get_name() + "::X");
  if (w->get_name().empty())
    w->set_name(conv_attr.get_name() + "::W");

  // Set inputs
  conv_attr.set_X(x).set_W(w);

  // Set outputs
  auto y = output_tensor(conv_attr.get_name() + "::Y");
  conv_attr.set_Y(y);

  // Create node and add to sub_nodes
  sub_nodes.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(conv_attr), context));

  return y;
}

} // namespace fusili

#endif // FUSILI_GRAPH_H
