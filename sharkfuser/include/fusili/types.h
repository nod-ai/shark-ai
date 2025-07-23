// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_TYPES_H
#define FUSILI_TYPES_H

namespace fusili {

enum class DataType {
  NotSet,
  Half,
  Float,
  Double,
  Int8,
  Int32,
  Int64,
  BFloat16,
  Boolean,
  FP8E4M3,
  FP8E5M2,
  FP4E2M1,
};

} // namespace fusili

#endif // FUSILI_TYPES_H
