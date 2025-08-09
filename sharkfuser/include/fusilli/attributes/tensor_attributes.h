// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the `TensorAttr` class definition for all compile-time
// constant metadata pertaining to tensors.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_TENSOR_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_TENSOR_ATTRIBUTES_H

#include "fusilli/attributes/types.h"
#include "fusilli/graph/context.h"
#include "fusilli/support/logging.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace fusilli {

// A note on memory layouts, strides and contiguity of tensors:
//
// There is no notion of strides in MLIR / IREE compiler. It expects the tensors
// to be contiguous (row-major) as specified by its shape.
//
// A contiguous tensor has the same logical and physical (in-memory) layout and
// doesn't require strides to define memory access patterns. But if we were to
// assign strides, a contiguous tensor of shape (n, c, h, w) may be accessed in
// row-major order using strides (c*h*w, h*w, w, 1) - reversed cumulative
// product of its dimensions. Likewise a contiguous tensor of shape (n, h, w, c)
// may be accessed in row-major order using strides (h*w*c, w*c, c, 1).
// Essentially, strides of contiguous row-major tensors are strictly
// monotonically decreasing (increasing for column major) and the last dimension
// (first for column major) always has a stride of 1.
//
// A non-contiguous tensor is one that has a different logical layout than
// the physical (in-memory) layout. For example, an NCHW (logical) tensor may be
// stored in memory as NHWC (physical) using a non-contiguous stride.
//
// Basically, strides can be used to represent different physical (in-memory)
// formats while preserving the same logical format for tensor layouts.
//
// PyTorch...
//
// Fusilli sits between the high level ML frameworks (PyTorch) and the low level
// MLIR compilers (IREE). Naturally it needs to account for these differences
// in tensor representation and provide a unified interface for both ends.
//
// With this in mind, here is Fusilli's memory layout convention:

// Logical layout is always "channels-first" (NCHW if 4D).
// Physical layout (in memory) is determined by the strides.
//
// T with logical shape (n,c,h,w) and stride (c*h*w, h*w, w, 1) is NCHW
// physical.
//
// T with logical shape (n,c,h,w) and stride (c*h*w, 1, c*w, c) is
// NHWC physical.
//

class TensorAttr {
public:
  using scalar_t = std::variant<int64_t, int32_t, float, double>;

  ErrorObject validate() const {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating tensor '" << name_ << "'");

    FUSILLI_RETURN_ERROR_IF(dim_.empty(), ErrorCode::AttributeNotSet,
                            "Tensor '" + name_ + "' dims not set");

    FUSILLI_RETURN_ERROR_IF(stride_.empty(), ErrorCode::AttributeNotSet,
                            "Tensor '" + name_ + "' strides not set");

    FUSILLI_RETURN_ERROR_IF(
        dim_.size() != stride_.size(), ErrorCode::InvalidAttribute,
        "Tensor '" + name_ +
            "' uses dim and stride of different dimensionality");

    FUSILLI_RETURN_ERROR_IF(dataType_ == DataType::NotSet,
                            ErrorCode::AttributeNotSet,
                            "Tensor '" + name_ + "' data type not set");

    FUSILLI_RETURN_ERROR_IF(
        isVirtual_ && isScalar_, ErrorCode::InvalidAttribute,
        "Tensor '" + name_ +
            "' cannot be both virtual (intermediate) and a scalar constant");

    FUSILLI_RETURN_ERROR_IF(
        scalarValue_.has_value() && !isScalar_, ErrorCode::InvalidAttribute,
        "Tensor '" + name_ +
            "' has a scalar value set but is not marked as a scalar");

    FUSILLI_RETURN_ERROR_IF(
        !scalarValue_.has_value() && isScalar_, ErrorCode::InvalidAttribute,
        "Tensor '" + name_ +
            "' is marked as a scalar but does not have a scalar value set");

    // Check for contiguity (inner dim stride is 1, monotonic)
    FUSILLI_RETURN_ERROR_IF(
        !(std::is_sorted(stride_.begin(), stride_.end(),
                         std::greater<int64_t>()) &&
          stride_.back() == 1),
        ErrorCode::NotImplemented,
        "Tensor '" + name_ +
            "' is not contiguous as defined by its stride; please specify a "
            "stride {A, B, ... Z} where A > B > ... Z and Z == 1. "
            "This will be supported in a future release");

    return ok();
  }

  TensorAttr() = default;

  // Constructors for scalar values
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

  // Fill datatypes from overall context when not set
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

  // MLIR assembly emitter helper methods
  std::string getValueTensorTypeAsm() const;
  std::string getMlirSSAValueNameAsm() const;

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
};

// Sorting function for deterministic lookups on TensorAttr containers
// (`std::set`) ensuring iteration orders are deterministic. It sorts
// by name.
struct TensorAttrSortByName {
  bool operator()(const std::shared_ptr<TensorAttr> &a,
                  const std::shared_ptr<TensorAttr> &b) const {
    return a->getName() < b->getName();
  }
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_TENSOR_ATTRIBUTES_H
