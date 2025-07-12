// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_TYPES_H
#define FUSILI_TYPES_H

namespace fusili {

enum class DataType_t {
  NOT_SET,

  HALF,
  FLOAT,
  DOUBLE,
  INT8,
  INT32,
  INT64,
  BFLOAT16,
  BOOLEAN,
  FP8_E4M3,
  FP8_E5M2,
  FP4_E2M1,
};

} // namespace fusili

#endif // FUSILI_TYPES_H
