// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains `AttributesCRTP` base class definitions which every node
// attribute derives from. It uses the CRTP pattern (aka F-bound polymorphism):
// https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
// It also contains macro definitions to easily populate generic getter/setter
// methods in the derived node attribute classes for fetching/setting input
// and output tensors.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_ATTRIBUTES_H
#define FUSILLI_ATTRIBUTES_ATTRIBUTES_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/context.h"
#include "fusilli/types.h"

#include <memory>

namespace fusilli {

// Every class that derives from AttributesCRTP should have two maps:
//   std::unordered_map<input_names, std::shared_ptr<TensorAttr>> inputs;
//   std::unordered_map<output_names, std::shared_ptr<TensorAttr>> outputs;
// These are used to populate metadata (e.g. data types) from the context,
// as well as have the macros auto-generate getters/setters for inputs/outputs.
template <typename DerivedT> class AttributesCRTP {
public:
  DataType computeDataType = DataType::NotSet;

  const std::string &getName() const { return name_; }

  DerivedT &setName(const std::string &name) {
    name_ = name;
    return self();
  }

  DerivedT &setComputeDataType(DataType type) {
    computeDataType = type;
    return self();
  }

  template <typename KeyT>
  DerivedT &setInput(KeyT key, const std::shared_ptr<TensorAttr> &tensor) {
    self().inputs[key] = tensor;
    return self();
  }

  template <typename KeyT>
  DerivedT &setOutput(KeyT key, const std::shared_ptr<TensorAttr> &tensor) {
    self().outputs[key] = tensor;
    return self();
  }

  template <typename KeyT>
  std::shared_ptr<TensorAttr> getInput(KeyT key) const {
    auto it = self().inputs.find(key);
    if (it != self().inputs.end()) {
      return it->second;
    }
    return nullptr;
  }

  template <typename KeyT>
  std::shared_ptr<TensorAttr> getOutput(KeyT key) const {
    auto it = self().outputs.find(key);
    if (it != self().outputs.end()) {
      return it->second;
    }
    return nullptr;
  }

  // Populate missing fields (e.g. datatypes) on the node and
  // tensor attributes from the graph context
  void fillFromContext(const Context &context) {
    if (computeDataType == DataType::NotSet) {
      setComputeDataType(context.getComputeDataType());
    }

    for (auto &kv : self().inputs) {
      auto &tensor = kv.second;
      if (tensor) {
        tensor->fillFromContext(context);
      }
    }

    for (auto &kv : self().outputs) {
      auto &tensor = kv.second;
      if (tensor) {
        tensor->fillFromContext(context);
      }
    }
  }

private:
  DerivedT &self() { return static_cast<DerivedT &>(*this); }
  const DerivedT &self() const { return static_cast<const DerivedT &>(*this); }
  std::string name_;
};

// Helper macros for generic input/output tensor getter/setter
#define FUSILLI_GENERIC_INPUT_TENSOR_GETTER(KTYPE, NAME)                       \
  std::shared_ptr<TensorAttr> get##NAME() const {                              \
    return getInput(KTYPE::NAME);                                              \
  }

#define FUSILLI_GENERIC_OUTPUT_TENSOR_GETTER(KTYPE, NAME)                      \
  std::shared_ptr<TensorAttr> get##NAME() const {                              \
    return getOutput(KTYPE::NAME);                                             \
  }

#define FUSILLI_GENERIC_INPUT_TENSOR_SETTER(RTYPE, KTYPE, NAME)                \
  RTYPE &set##NAME(const std::shared_ptr<TensorAttr> &tensor) {                \
    return setInput(KTYPE::NAME, tensor);                                      \
  }

#define FUSILLI_GENERIC_OUTPUT_TENSOR_SETTER(RTYPE, KTYPE, NAME)               \
  RTYPE &set##NAME(const std::shared_ptr<TensorAttr> &tensor) {                \
    return setOutput(KTYPE::NAME, tensor);                                     \
  }

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_ATTRIBUTES_H
