// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains attributes (compile-time constant metadata) for
// matrix multiplication nodes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_MATMUL_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_MATMUL_ATTRIBUTES_H

#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace fusilli {

class MatmulAttr : public AttributesCRTP<MatmulAttr> {
public:
  // Names for Tensor Inputs and Outputs (doesn't include constant attributes).
  enum class InputNames : uint8_t { A, B };
  enum class OutputNames : uint8_t { C };

  std::unordered_map<InputNames, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<OutputNames, std::shared_ptr<TensorAttr>> outputs;

  // Setters:
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(MatmulAttr, InputNames, A)
  FUSILLI_GENERIC_INPUT_TENSOR_SETTER(MatmulAttr, InputNames, B)
  FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(MatmulAttr, OutputNames, C)

  // Getters:
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, A)
  FUSILLI_GENERIC_INPUT_TENSOR_GETTER(InputNames, B)
  FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(OutputNames, C)
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_MATMUL_ATTRIBUTES_H
