// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the element types used throughout Fusilli datastructures.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_TYPES_H
#define FUSILLI_TYPES_H

#include <string>
#include <unordered_map>

namespace fusilli {

enum class DataType {
  NotSet,
  Half,
  BFloat16,
  Float,
  Double,
  Uint8,
  Int8,
  Int16,
  Int32,
  Int64,
  Boolean,
  FP8E5M2,
};

} // namespace fusilli

#endif // FUSILLI_TYPES_H
