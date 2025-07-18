// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_CONTEXT_H
#define FUSILI_CONTEXT_H

#include "fusili/types.h"

#include <string>

namespace fusili {

class Context {
public:
  // Setters
  Context &setIntermediateDataType(DataType_t type) {
    intermediateDataType_ = type;
    return *this;
  }

  Context &setIODataType(DataType_t type) {
    ioDataType_ = type;
    return *this;
  }

  Context &setComputeDataType(DataType_t type) {
    computeDataType_ = type;
    return *this;
  }

  Context &setName(const std::string &name) {
    name_ = name;
    return *this;
  }

  // Getters
  DataType_t getIODataType() const { return ioDataType_; }

  DataType_t getIntermediateDataType() const { return intermediateDataType_; }

  DataType_t getComputeDataType() const { return computeDataType_; }

  const std::string &getName() const { return name_; }

private:
  DataType_t computeDataType_ = DataType_t::NOT_SET;
  DataType_t intermediateDataType_ = DataType_t::NOT_SET;
  DataType_t ioDataType_ = DataType_t::NOT_SET;
  std::string name_;
};

} // namespace fusili

#endif // FUSILI_CONTEXT_H
