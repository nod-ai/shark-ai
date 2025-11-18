// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the `Context` class which can be thought
// of as global attributes for the Graph. This is useful in populating missing
// metadata on tensors and/or nodes in the graph among other things.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_CONTEXT_H
#define FUSILLI_GRAPH_CONTEXT_H

#include "fusilli/attributes/types.h"

#include <string>

namespace fusilli {

class Context {
public:
  // Setters:
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

  // Getters:
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

} // namespace fusilli

#endif // FUSILLI_GRAPH_CONTEXT_H
