// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "fusili/attributes/tensor_attributes.h"

namespace fusili {

template <typename DerivedT> class AttributesCRTP {
private:
  DerivedT &self() { return static_cast<DerivedT &>(*this); }
  const DerivedT &self() const { return static_cast<const DerivedT &>(*this); }

public:
  std::string name;

  DerivedT &set_name(const std::string &name_) {
    name = name_;
    return self();
  }
};

} // namespace fusili
