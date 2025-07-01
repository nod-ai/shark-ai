// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <unordered_set>

#include "context.h"

namespace fusili {

class Graph {
private:
  Context context;
  std::unordered_set<std::shared_ptr<TensorAttr>> full_graph_inputs;
  std::unordered_set<std::shared_ptr<TensorAttr>> full_graph_outputs;

public:
  Graph &set_io_data_type(DataType_t const type);

  Graph &set_compute_data_type(DataType_t const type);

  Graph &set_intermediate_data_type(DataType_t const type);

  error_t query_tensor_of_uid(int64_t const uid, TensorAttr &tensor) const;

  std::shared_ptr<TensorAttr> tensor(TensorAttr const &tensor);
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
  full_graph_inputs.emplace(tensor_ptr);
  return tensor_ptr;
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
