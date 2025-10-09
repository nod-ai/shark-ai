// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for fusilli tests.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_TESTS_UTILS_H
#define FUSILLI_TESTS_UTILS_H

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

// Unwrap the type returned from an expression that evaluates to an ErrorOr,
// fail the test using Catch2's REQUIRE if the result is an ErrorObject.
//
// This is very similar to FUSILLI_TRY, but FUSILLI_TRY propagates an error to
// callers on the error path, this fails the test on the error path. The two
// macros are analogous to rust's `?` (try) operator and `.unwrap()` call.
#define FUSILLI_REQUIRE_UNWRAP(expr)                                           \
  ({                                                                           \
    auto _errorOr = (expr);                                                    \
    REQUIRE(isOk(_errorOr));                                                   \
    std::move(*_errorOr);                                                      \
  })

// Utility to convert vector of dims from int64_t to size_t (unsigned long)
// which is compatible with `iree_hal_dim_t` and fixes narrowing conversion
// warnings.
inline std::vector<size_t> castToSizeT(const std::vector<int64_t> &input) {
  return std::vector<size_t>(input.begin(), input.end());
}

namespace fusilli {

inline ErrorOr<std::shared_ptr<Buffer>>
allocateBufferOfType(Handle &handle, const std::vector<int64_t> &shape,
                     int64_t volume, DataType type, float initVal) {
  switch (type) {
  case DataType::Half:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(shape),
        /*bufferData=*/std::vector<half>(volume, half(initVal)))));
  case DataType::BFloat16:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(shape),
        /*bufferData=*/std::vector<bf16>(volume, bf16(initVal)))));
  case DataType::Float:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(shape),
        /*bufferData=*/std::vector<float>(volume, float(initVal)))));
  default:
    return error(ErrorCode::InvalidAttribute, "Unsupported DataType");
  }
}

} // namespace fusilli

#endif // FUSILLI_TESTS_UTILS_H
