// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_ATTRIBUTES_CONV_ATTRIBUTES_H
#define FUSILI_ATTRIBUTES_CONV_ATTRIBUTES_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "fusili/attributes/attributes.h"
#include "fusili/attributes/tensor_attributes.h"

namespace fusili {

class ConvFPropAttr : public AttributesCRTP<ConvFPropAttr> {
private:
  std::vector<int64_t> pre_padding;
  std::vector<int64_t> post_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;

public:
  enum class input_names { X, W };
  enum class output_names { Y };

  std::unordered_map<input_names, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<output_names, std::shared_ptr<TensorAttr>> outputs;

  // Setters
  FUSILI_GENERIC_INPUT_TENSOR_SETTER(ConvFPropAttr, input_names, X)

  FUSILI_GENERIC_INPUT_TENSOR_SETTER(ConvFPropAttr, input_names, W)

  FUSILI_GENERIC_OUTPUT_TENSOR_SETTER(ConvFPropAttr, output_names, Y)

  ConvFPropAttr &set_pre_padding(std::vector<int64_t> const &padding) {
    pre_padding = padding;
    return *this;
  }

  ConvFPropAttr &set_post_padding(std::vector<int64_t> const &padding) {
    post_padding = padding;
    return *this;
  }

  ConvFPropAttr &set_padding(std::vector<int64_t> const &padding) {
    pre_padding = padding;
    post_padding = padding;
    return *this;
  }

  ConvFPropAttr &set_stride(std::vector<int64_t> const &stride_) {
    stride = stride_;
    return *this;
  }

  ConvFPropAttr &set_dilation(std::vector<int64_t> const &dilation_) {
    dilation = dilation_;
    return *this;
  }

  // Getters
  FUSILI_GENERIC_INPUT_TENSOR_GETTER(input_names, X)

  FUSILI_GENERIC_INPUT_TENSOR_GETTER(input_names, W)

  FUSILI_GENERIC_OUTPUT_TENSOR_GETTER(output_names, Y)

  const std::vector<int64_t> &get_pre_padding() const { return pre_padding; }

  const std::vector<int64_t> &get_post_padding() const { return post_padding; }

  const std::vector<int64_t> &get_stride() const { return stride; }

  const std::vector<int64_t> &get_dilation() const { return dilation; }
};

} // namespace fusili

#endif // FUSILI_ATTRIBUTES_CONV_ATTRIBUTES_H
