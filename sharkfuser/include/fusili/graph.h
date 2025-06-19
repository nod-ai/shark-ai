// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>

#include "context.h"

namespace fusili {
namespace graph {

class Graph {
private:
  Context context;

public:
  inline Graph &set_io_data_type(DataType_t const type) {
    context.set_io_data_type(type);
    return *this;
  }

  inline Graph &set_compute_data_type(DataType_t const type) {
    context.set_compute_data_type(type);
    return *this;
  }

  inline Graph &set_intermediate_data_type(DataType_t const type) {
    context.set_intermediate_data_type(type);
    return *this;
  }
};

} // namespace graph
} // namespace fusili
