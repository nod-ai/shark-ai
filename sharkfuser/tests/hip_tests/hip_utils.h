// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILLI_TESTS_HIP_TESTS_HIP_UTILS_H
#define FUSILLI_TESTS_HIP_TESTS_HIP_UTILS_H

// Utility macro to check status of HIP functions that are set with nodiscard.
#define HIP_REQUIRE_SUCCESS(expr)                                              \
  ({                                                                           \
    auto err = (expr);                                                         \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "Error: %s\n", hipGetErrorString(err));                  \
    }                                                                          \
    REQUIRE(err == hipSuccess);                                                \
  })

#endif // FUSILLI_TESTS_HIP_TESTS_HIP_UTILS_H
