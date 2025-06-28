// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace fusili {

enum class DataType_t {
  NOT_SET,

  FLOAT,
  DOUBLE,
  HALF,
  INT8,
  INT32,
  INT8x4,
  UINT8,
  UINT8x4,
  INT8x32,
  BFLOAT16,
  INT64,
  BOOLEAN,
  FP8_E4M3,
  FP8_E5M2,
  FAST_FLOAT_FOR_FP8,
  FP8_E8M0,
  FP4_E2M1,
};

} // namespace fusili
