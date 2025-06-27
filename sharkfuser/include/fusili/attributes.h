// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "logging.h"
#include "types.h"

namespace fusili {

class TensorAttr {
public:
  using uid_t = int64_t;

  error_t validate() const {
    FUSILI_RETURN_ERROR_IF(dim.empty(), error_code_t::ATTRIBUTE_NOT_SET,
                           "Tensor '" + name + "' dims not set");
    FUSILI_RETURN_ERROR_IF(stride.empty(), error_code_t::ATTRIBUTE_NOT_SET,
                           "Tensor '" + name + "' strides not set");
    FUSILI_RETURN_ERROR_IF(
        dim.size() != stride.size(), error_code_t::INVALID_ATTRIBUTE,
        "Tensor '" + name +
            "' uses dim and stride of different dimensionality");

    return {error_code_t::OK, ""};
  }

private:
  std::string name;
  DataType_t data_type = DataType_t::NOT_SET;
  std::vector<int64_t> dim = {};
  std::vector<int64_t> stride = {};

  // Intermediate tensors that are not inputs/outputs are virtual
  // and not stored/read as they appear internal to the kernel.
  // They also don't need their shapes and sizes specified.
  bool is_virtual = false;

  uid_t uid = 0;
  bool uid_assigned = false;
};

} // namespace fusili
