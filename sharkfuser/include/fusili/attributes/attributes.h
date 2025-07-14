// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_ATTRIBUTES_ATTRIBUTES_H
#define FUSILI_ATTRIBUTES_ATTRIBUTES_H

#include "fusili/attributes/tensor_attributes.h"

namespace fusili {

// Every class that derives from AttributeCRTP should have two maps:
//  std::unordered_map<input_names, std::shared_ptr<TensorAttr>> inputs;
//  std::unordered_map<output_names, std::shared_ptr<TensorAttr>> outputs;
// These are used to populate metadata (e.g. data types) from the context.
template <typename DerivedT> class AttributesCRTP {
private:
  DerivedT &self() { return static_cast<DerivedT &>(*this); }
  const DerivedT &self() const { return static_cast<const DerivedT &>(*this); }

  std::string name;

public:
  DataType_t compute_data_type = DataType_t::NOT_SET;

  const std::string &get_name() const { return name; }

  DerivedT &set_name(std::string const &name_) {
    name = name_;
    return self();
  }

  DerivedT &set_compute_data_type(DataType_t const value) {
    compute_data_type = value;
    return self();
  }

  template <typename KeyT>
  DerivedT &set_input(KeyT key, std::shared_ptr<TensorAttr> const &tensor) {
    self().inputs[key] = tensor;
    return self();
  }

  template <typename KeyT>
  DerivedT &set_output(KeyT key, std::shared_ptr<TensorAttr> const &tensor) {
    self().outputs[key] = tensor;
    return self();
  }

  template <typename KeyT>
  std::shared_ptr<TensorAttr> get_input(KeyT key) const {
    auto it = self().inputs.find(key);
    if (it != self().inputs.end()) {
      return it->second;
    }
    return nullptr;
  }

  template <typename KeyT>
  std::shared_ptr<TensorAttr> get_output(KeyT key) const {
    auto it = self().outputs.find(key);
    if (it != self().outputs.end()) {
      return it->second;
    }
    return nullptr;
  }

  void fill_from_context(Context const &context) {
    if (compute_data_type == DataType_t::NOT_SET) {
      set_compute_data_type(context.get_compute_data_type());
    }

    for (auto &[_, tensor] : self().inputs) {
      if (tensor)
        tensor->fill_from_context(context);
    }

    for (auto &[_, tensor] : self().outputs) {
      if (tensor)
        tensor->fill_from_context(context);
    }
  }
};

#define FUSILI_GENERIC_INPUT_TENSOR_GETTER(KTYPE, NAME)                        \
  std::shared_ptr<TensorAttr> get_##NAME() const {                             \
    return get_input(KTYPE::NAME);                                             \
  }

#define FUSILI_GENERIC_OUTPUT_TENSOR_GETTER(KTYPE, NAME)                       \
  std::shared_ptr<TensorAttr> get_##NAME() const {                             \
    return get_output(KTYPE::NAME);                                            \
  }

#define FUSILI_GENERIC_INPUT_TENSOR_SETTER(RTYPE, KTYPE, NAME)                 \
  RTYPE &set_##NAME(std::shared_ptr<TensorAttr> const &tensor) {               \
    return set_input(KTYPE::NAME, tensor);                                     \
  }

#define FUSILI_GENERIC_OUTPUT_TENSOR_SETTER(RTYPE, KTYPE, NAME)                \
  RTYPE &set_##NAME(std::shared_ptr<TensorAttr> const &tensor) {               \
    return set_output(KTYPE::NAME, tensor);                                    \
  }

} // namespace fusili

#endif // FUSILI_ATTRIBUTES_ATTRIBUTES_H
