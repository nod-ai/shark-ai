// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_ATTRIBUTES_TENSOR_ATTRIBUTES_H
#define FUSILI_ATTRIBUTES_TENSOR_ATTRIBUTES_H

#include "fusili/context.h"
#include "fusili/logging.h"
#include "fusili/types.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace fusili {

class TensorAttr {
public:
  using uid_t = int64_t;
  using scalar_t = std::variant<int64_t, int32_t, float, double>;

  error_t validate() const {
    FUSILI_RETURN_ERROR_IF(dim_.empty(), error_code_t::AttributeNotSet,
                           "Tensor '" + name_ + "' dims not set");
    FUSILI_RETURN_ERROR_IF(stride_.empty(), error_code_t::AttributeNotSet,
                           "Tensor '" + name_ + "' strides not set");
    FUSILI_RETURN_ERROR_IF(
        dim_.size() != stride_.size(), error_code_t::InvalidAttribute,
        "Tensor '" + name_ +
            "' uses dim and stride of different dimensionality");

    FUSILI_RETURN_ERROR_IF(
        isVirtual_ && isScalar_, error_code_t::InvalidAttribute,
        "Tensor '" + name_ +
            "' cannot be both virtual (intermediate) and a scalar constant");

    FUSILI_RETURN_ERROR_IF(
        scalarValue_.has_value() && !isScalar_, error_code_t::InvalidAttribute,
        "Tensor '" + name_ +
            "' has a scalar value set but is not marked as a scalar");

    FUSILI_RETURN_ERROR_IF(
        !scalarValue_.has_value() && isScalar_, error_code_t::InvalidAttribute,
        "Tensor '" + name_ +
            "' is marked as a scalar but does not have a scalar value set");

    // Check for contiguity (inner dim stride is 1, monotonic)
    FUSILI_RETURN_ERROR_IF(
        !(std::is_sorted(stride_.begin(), stride_.end(),
                         std::greater<int64_t>()) &&
          stride_.back() == 1),
        error_code_t::NotImplemented,
        "Tensor '" + name_ +
            "' is not contiguous as defined by its stride; please specify a "
            "stride {A, B, ... Z} where A > B > ... Z and Z == 1. "
            "This will be supported in a future release");

    return {error_code_t::OK, ""};
  }

  TensorAttr() = default;

  explicit TensorAttr(float value) {
    scalarValue_ = value;
    isScalar_ = true;
    dim_ = stride_ = {1};
    dataType_ = DataType::Float;
  }

  explicit TensorAttr(double value) {
    scalarValue_ = value;
    isScalar_ = true;
    dim_ = stride_ = {1};
    dataType_ = DataType::Double;
  }

  explicit TensorAttr(int32_t value) {
    scalarValue_ = value;
    isScalar_ = true;
    dim_ = stride_ = {1};
    dataType_ = DataType::Int32;
  }

  explicit TensorAttr(int64_t value) {
    scalarValue_ = value;
    isScalar_ = true;
    dim_ = stride_ = {1};
    dataType_ = DataType::Int64;
  }

  TensorAttr &fillFromContext(const Context &context) {
    if (getDataType() == DataType::NotSet) {
      if (isVirtual()) {
        setDataType(context.getIntermediateDataType());
      } else {
        setDataType(context.getIODataType());
      }
    }
    return *this;
  }

  // Setters
  TensorAttr &setName(const std::string &value) {
    name_ = value;
    return *this;
  }

  TensorAttr &setDataType(DataType value) {
    dataType_ = value;
    return *this;
  }

  TensorAttr &setDim(const std::vector<int64_t> &value) {
    dim_ = value;
    return *this;
  }

  TensorAttr &setStride(const std::vector<int64_t> &value) {
    stride_ = value;
    return *this;
  }

  TensorAttr &setIsVirtual(bool value) {
    isVirtual_ = value;
    return *this;
  }

  TensorAttr &setOutput(bool value) { return setIsVirtual(!value); }

  TensorAttr &setIsScalar(bool value) {
    isScalar_ = value;
    return *this;
  }

  TensorAttr &setUid(uid_t value) {
    uid_ = value;
    uidSet_ = true;
    return *this;
  }

  TensorAttr &clearUid() {
    uid_ = 0;
    uidSet_ = false;
    return *this;
  }

  // Getters
  const std::string &getName() const { return name_; }

  DataType getDataType() const { return dataType_; }

  const std::vector<int64_t> &getDim() const { return dim_; }

  const std::vector<int64_t> &getStride() const { return stride_; }

  int64_t getVolume() const {
    int64_t volume = 1;
    for (const auto &d : dim_) {
      volume *= d;
    }
    return volume;
  }

  bool isVirtual() const { return isVirtual_; }

  bool isScalar() const { return isScalar_; }

  std::optional<scalar_t> getScalarValue() const { return scalarValue_; }

  uid_t getUid() const { return uid_; }

  bool hasUid() const { return uidSet_; }

private:
  std::string name_;
  DataType dataType_ = DataType::NotSet;
  std::vector<int64_t> dim_ = {};
  std::vector<int64_t> stride_ = {};

  // Intermediate tensors that are not inputs/outputs are virtual
  // and not stored/read as they appear internal to the kernel.
  // They also don't need their shapes and sizes specified.
  bool isVirtual_ = false;

  // To represent scalar constants either obtained through
  // constant folding, or passed in as scalars during execution
  bool isScalar_ = false;
  std::optional<scalar_t> scalarValue_ = std::nullopt;

  // Unique identifier for every tensor in the graph
  uid_t uid_ = 0;
  bool uidSet_ = false;
};

} // namespace fusili

#endif // FUSILI_ATTRIBUTES_TENSOR_ATTRIBUTES_H
