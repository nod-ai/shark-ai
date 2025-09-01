// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for fusilli samples.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SAMPLES_UTILS_H
#define FUSILLI_SAMPLES_UTILS_H

#include <cstddef>
#include <cstdint>
#include <vector>

// Utility to convert vector of dims from int64_t to size_t (unsigned long)
// which is compatible with `iree_hal_dim_t` and fixes narrowing conversion
// warnings.
inline std::vector<size_t> castToSizeT(const std::vector<int64_t> &input) {
  return std::vector<size_t>(input.begin(), input.end());
}

#endif // FUSILLI_SAMPLES_UTILS_H
