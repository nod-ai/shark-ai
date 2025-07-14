// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_ATTRIBUTES_TENSOR_ATTRIBUTES_H
#define FUSILI_ATTRIBUTES_TENSOR_ATTRIBUTES_H

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "fusili/context.h"
#include "fusili/logging.h"
#include "fusili/types.h"

namespace fusili {

class TensorAttr {
public:
  using uid_t = int64_t;
  using scalar_t = std::variant<int64_t, int32_t, float, double>;

  error_t validate() const {
    FUSILI_RETURN_ERROR_IF(dim.empty(), error_code_t::ATTRIBUTE_NOT_SET,
                           "Tensor '" + name + "' dims not set");
    FUSILI_RETURN_ERROR_IF(stride.empty(), error_code_t::ATTRIBUTE_NOT_SET,
                           "Tensor '" + name + "' strides not set");
    FUSILI_RETURN_ERROR_IF(
        dim.size() != stride.size(), error_code_t::INVALID_ATTRIBUTE,
        "Tensor '" + name +
            "' uses dim and stride of different dimensionality");

    FUSILI_RETURN_ERROR_IF(
        is_virtual && is_scalar, error_code_t::INVALID_ATTRIBUTE,
        "Tensor '" + name +
            "' cannot be both virtual (intermediate) and a scalar constant");

    FUSILI_RETURN_ERROR_IF(
        scalar_value.has_value() && !is_scalar, error_code_t::INVALID_ATTRIBUTE,
        "Tensor '" + name +
            "' has a scalar value set but is not marked as a scalar");

    FUSILI_RETURN_ERROR_IF(
        !scalar_value.has_value() && is_scalar, error_code_t::INVALID_ATTRIBUTE,
        "Tensor '" + name +
            "' is marked as a scalar but does not have a scalar value set");

    return {error_code_t::OK, ""};
  }

private:
  std::string name;
  DataType_t data_type = DataType_t::NOT_SET;
  std::vector<int64_t> dim = {};
  std::vector<int64_t> stride = {};

  // Intermediate tensors that are not inputs/outputs are virtual
  // and not stored/read as they appear internal to the kernel.
  // They also don't need their shapes and sizes specified.
  bool is_virtual = false;

  // To represent scalar constants either obtained through
  // constant folding, or passed in as scalars during execution
  bool is_scalar = false;
  std::optional<scalar_t> scalar_value = std::nullopt;

  // Unique identifier for every tensor in the graph
  uid_t uid = 0;
  bool uid_set = false;

public:
  TensorAttr() = default;

  TensorAttr(float const &value) {
    scalar_value = value;
    is_scalar = true;
    dim = stride = {1};
    data_type = DataType_t::FLOAT;
  }

  TensorAttr(double const &value) {
    scalar_value = value;
    is_scalar = true;
    dim = stride = {1};
    data_type = DataType_t::DOUBLE;
  }

  TensorAttr(int32_t const &value) {
    scalar_value = value;
    is_scalar = true;
    dim = stride = {1};
    data_type = DataType_t::INT32;
  }

  TensorAttr(int64_t const &value) {
    scalar_value = value;
    is_scalar = true;
    dim = stride = {1};
    data_type = DataType_t::INT64;
  }

  TensorAttr &fill_from_context(Context const &context) {
    if (get_data_type() == DataType_t::NOT_SET) {
      if (get_is_virtual())
        set_data_type(context.get_intermediate_data_type());
      else
        set_data_type(context.get_io_data_type());
    }
    return *this;
  }

  // Setters
  TensorAttr &set_name(std::string const &value) {
    name = value;
    return *this;
  }

  TensorAttr &set_data_type(DataType_t const value) {
    data_type = value;
    return *this;
  }

  TensorAttr &set_dim(std::vector<int64_t> const &value) {
    dim = value;
    return *this;
  }

  TensorAttr &set_stride(std::vector<int64_t> const &value) {
    stride = value;
    return *this;
  }

  TensorAttr &set_is_virtual(bool const value) {
    is_virtual = value;
    return *this;
  }

  TensorAttr &set_output(bool const value) { return set_is_virtual(!value); }

  TensorAttr &set_is_scalar(bool const value) {
    is_scalar = value;
    return *this;
  }

  TensorAttr &set_uid(uid_t const value) {
    uid = value;
    uid_set = true;
    return *this;
  }

  TensorAttr &clear_uid() {
    uid = 0;
    uid_set = false;
    return *this;
  }

  // Getters
  const std::string &get_name() const { return name; }

  DataType_t get_data_type() const { return data_type; }

  const std::vector<int64_t> &get_dim() const { return dim; }

  const std::vector<int64_t> &get_stride() const { return stride; }

  int64_t get_volume() const {
    int64_t volume = 1;
    for (const int64_t &d : dim)
      volume *= d;
    return volume;
  }

  bool get_is_virtual() const { return is_virtual; }

  bool get_is_scalar() const { return is_scalar; }

  std::optional<scalar_t> get_scalar_value() const { return scalar_value; }

  uid_t get_uid() const { return uid; }

  bool has_uid() const { return uid_set; }
};

} // namespace fusili

#endif // FUSILI_ATTRIBUTES_TENSOR_ATTRIBUTES_H
