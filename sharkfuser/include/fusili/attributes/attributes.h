// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "tensor_attributes.h"

namespace fusili {

// Every class that derives from AttributeCRTP should have two maps:
//  std::unordered_map<input_names, std::shared_ptr<TensorAttr>> inputs;
//  std::unordered_map<output_names, std::shared_ptr<TensorAttr>> outputs;
// These are used to populate metadata (e.g. data types) from the context.
template <typename DerivedT> class AttributesCRTP {
private:
  DerivedT &self() { return static_cast<DerivedT &>(*this); }
  const DerivedT &self() const { return static_cast<const DerivedT &>(*this); }

public:
  std::string name;
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

} // namespace fusili
