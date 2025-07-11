// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_CONTEXT_H
#define FUSILI_CONTEXT_H

#include <string>

#include "fusili/types.h"

namespace fusili {

class Context {
private:
  DataType_t compute_data_type = DataType_t::NOT_SET;
  DataType_t intermediate_data_type = DataType_t::NOT_SET;
  DataType_t io_data_type = DataType_t::NOT_SET;

  std::string name = "";

public:
  // Setters
  Context &set_intermediate_data_type(DataType_t const type) {
    intermediate_data_type = type;
    return *this;
  }

  Context &set_io_data_type(DataType_t const type) {
    io_data_type = type;
    return *this;
  }

  Context &set_compute_data_type(DataType_t const type) {
    compute_data_type = type;
    return *this;
  }

  Context &set_name(std::string const &name_) {
    name = name_;
    return *this;
  }

  // Getters
  DataType_t get_io_data_type() const { return io_data_type; }

  DataType_t get_intermediate_data_type() const {
    return intermediate_data_type;
  }

  DataType_t get_compute_data_type() const { return compute_data_type; }

  std::string get_name() const { return name; }
};

} // namespace fusili

#endif // FUSILI_CONTEXT_H
