// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// pointwise nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace fusilli {

class PointwiseAttr : public AttributesCRTP<PointwiseAttr> {
public:
  // Names for Tensor Inputs and Outputs. Pointwise can have a maximum of three
  // inputs.
  enum class InputNames { IN_0, IN_1, IN_2 };
  enum class OutputNames { OUT_0 };

  enum class Mode {
    NOT_SET,
    ADD,
    RELU_FWD,
  };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_0)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_1)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_2)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(PointwiseAttr, OutputNames, OUT_0)

  PointwiseAttr &setMode(Mode mode) {
    mode_ = mode;
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_0)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_1)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_2)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, OUT_0)

  Mode getMode() const { return mode_; }

  // Utilities for pointwise modes.
  static const std::unordered_map<Mode, std::string> modeToStr;
  static const std::unordered_map<PointwiseAttr::Mode, int>
      modeToRequiredInputCount;

private:
  Mode mode_ = Mode::NOT_SET;
};

inline const std::unordered_map<PointwiseAttr::Mode, std::string>
    PointwiseAttr::modeToStr = {
        {PointwiseAttr::Mode::NOT_SET, "NOT_SET"},
        {PointwiseAttr::Mode::RELU_FWD, "RELU_FWD"},
        {PointwiseAttr::Mode::ADD, "ADD"},
};
inline const std::unordered_map<PointwiseAttr::Mode, int>
    PointwiseAttr::modeToRequiredInputCount = {
        {PointwiseAttr::Mode::RELU_FWD, 1}, {PointwiseAttr::Mode::ADD, 2}};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H
