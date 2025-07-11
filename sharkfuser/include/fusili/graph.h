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

    return {error_code_t::OK, ""};
  }

  Type getType() override { return Type::COMPOSITE; }

  Graph &set_io_data_type(DataType_t const type);

  Graph &set_compute_data_type(DataType_t const type);

  Graph &set_intermediate_data_type(DataType_t const type);

  error_t query_tensor_of_uid(int64_t const uid, TensorAttr &tensor) const;

  std::shared_ptr<TensorAttr> tensor(TensorAttr const &tensor);

  std::shared_ptr<TensorAttr> conv_fprop(std::shared_ptr<TensorAttr> const &x,
                                         std::shared_ptr<TensorAttr> const &w,
                                         ConvFPropAttr &attributes);
};

inline Graph &Graph::set_io_data_type(DataType_t const type) {
  context.set_io_data_type(type);
  return *this;
}

inline Graph &Graph::set_compute_data_type(DataType_t const type) {
  context.set_compute_data_type(type);
  return *this;
}

inline Graph &Graph::set_intermediate_data_type(DataType_t const type) {
  context.set_intermediate_data_type(type);
  return *this;
}

inline std::shared_ptr<TensorAttr> Graph::tensor(TensorAttr const &tensor) {
  auto tensor_ptr = std::make_shared<TensorAttr>(tensor);
  full_graph_inputs.insert(tensor_ptr);
  return tensor_ptr;
}

inline std::shared_ptr<TensorAttr>
Graph::conv_fprop(std::shared_ptr<TensorAttr> const &x,
                  std::shared_ptr<TensorAttr> const &w, ConvFPropAttr &attr) {

  // Set inputs
  attr.set_X(x).set_W(w);

  // Set outputs
  if (attr.get_name().empty())
    attr.set_name("conv_fprop_" + std::to_string(sub_nodes.size()));
  auto y = output_tensor(attr.get_name() + "::Y");
  attr.set_Y(y);

  sub_nodes.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(attr), context));

  return y;
}

inline error_t Graph::query_tensor_of_uid(int64_t const uid,
                                          TensorAttr &tensor) const {
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

} // namespace fusili

#endif // FUSILI_GRAPH_H
