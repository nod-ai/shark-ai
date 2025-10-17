// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// convolution nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_CONV_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_CONV_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fusilli {

class ConvFPropAttr : public AttributesCRTP<ConvFPropAttr> {
public:
  // Names for Tensor Inputs and Outputs (doesn't include constant attributes).
  enum class InputNames { X, W };
  enum class OutputNames { Y };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(ConvFPropAttr, InputNames, X)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(ConvFPropAttr, InputNames, W)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(ConvFPropAttr, OutputNames, Y)

  ConvFPropAttr &setPadding(const std::vector<int64_t> &padding) {
    padding_ = padding;
    return *this;
  }
  template <Int64Range R> ConvFPropAttr &setPadding(R &&padding) {
    padding_.assign(padding.begin(), padding.end());
    return *this;
  }

  ConvFPropAttr &setStride(const std::vector<int64_t> &stride) {
    stride_ = stride;
    return *this;
  }
  template <Int64Range R> ConvFPropAttr &setStride(R &&stride) {
    stride_.assign(stride.begin(), stride.end());
    return *this;
  }

  ConvFPropAttr &setDilation(const std::vector<int64_t> &dilation) {
    dilation_ = dilation;
    return *this;
  }
  template <Int64Range R> ConvFPropAttr &setDilation(R &&dilation) {
    dilation_.assign(dilation.begin(), dilation.end());
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, X)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, W)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, Y)

  const std::vector<int64_t> &getPadding() const { return padding_; }
  const std::vector<int64_t> &getStride() const { return stride_; }
  const std::vector<int64_t> &getDilation() const { return dilation_; }

private:
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
};

class ConvWGradAttr : public AttributesCRTP<ConvWGradAttr> {
public:
  enum class InputNames { DY, X };
  enum class OutputNames { DW };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(ConvWGradAttr, InputNames, DY)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(ConvWGradAttr, InputNames, X)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(ConvWGradAttr, OutputNames, DW)

  ConvWGradAttr &setPadding(const std::vector<int64_t> &padding) {
    padding_ = padding;
    return *this;
  }
  template <Int64Range R> ConvWGradAttr &setPadding(R &&padding) {
    padding_.assign(padding.begin(), padding.end());
    return *this;
  }

  ConvWGradAttr &setStride(const std::vector<int64_t> &stride) {
    stride_ = stride;
    return *this;
  }
  template <Int64Range R> ConvWGradAttr &setStride(R &&stride) {
    stride_.assign(stride.begin(), stride.end());
    return *this;
  }

  ConvWGradAttr &setDilation(const std::vector<int64_t> &dilation) {
    dilation_ = dilation;
    return *this;
  }
  template <Int64Range R> ConvWGradAttr &setDilation(R &&dilation) {
    dilation_.assign(dilation.begin(), dilation.end());
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, DY)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, X)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, DW)

  const std::vector<int64_t> &getPadding() const { return padding_; }
  const std::vector<int64_t> &getStride() const { return stride_; }
  const std::vector<int64_t> &getDilation() const { return dilation_; }

private:
  std::vector<int64_t> padding_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> dilation_;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_CONV_ATTRIBUTES_H
