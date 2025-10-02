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

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fusilli {

class PointwiseAttr : public AttributesCRTP<PointwiseAttr> {
public:
  // Names for Tensor Inputs and Outputs. Pointwise can have a maximum of three
  // inputs.
  enum class InputNames { IN_0, IN_1, IN_2 };
  enum class OutputNames { OUT };

  enum class Mode {
    NOT_SET,
    ADD,
    RELU,
  };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_0)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_1)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(PointwiseAttr, InputNames, IN_2)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(PointwiseAttr, OutputNames, OUT)

  PointwiseAttr &setMode(Mode mode) {
    mode_ = mode;
    return *this;
  }

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_0)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_1)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, IN_2)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, OUT)

  Mode getMode() const { return mode_; }

  // Utility function to convert enum to string
  static std::string modeToString(Mode mode) {
    switch (mode) {
    case Mode::RELU:
      return "RELU";
    case Mode::ADD:
      return "ADD";
    default:
      return "UNKNOWN";
    }
  }

private:
  Mode mode_ = Mode::NOT_SET;
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_POINTWISE_ATTRIBUTES_H
