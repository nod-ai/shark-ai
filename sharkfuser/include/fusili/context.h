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
  Context &setIntermediateDataType(DataType type) {
    intermediateDataType_ = type;
    return *this;
  }

  Context &setIODataType(DataType type) {
    ioDataType_ = type;
    return *this;
  }

  Context &setComputeDataType(DataType type) {
    computeDataType_ = type;
    return *this;
  }

  Context &setName(const std::string &name) {
    name_ = name;
    return *this;
  }

  // Getters
  DataType getIODataType() const { return ioDataType_; }

  DataType getIntermediateDataType() const { return intermediateDataType_; }

  DataType getComputeDataType() const { return computeDataType_; }

  const std::string &getName() const { return name_; }

private:
  DataType computeDataType_ = DataType::NotSet;
  DataType intermediateDataType_ = DataType::NotSet;
  DataType ioDataType_ = DataType::NotSet;
  std::string name_;
};

} // namespace fusili

#endif // FUSILI_CONTEXT_H
