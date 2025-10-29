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

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_TESTS_UTILS_H
#define FUSILLI_TESTS_UTILS_H

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <utility> // IWYU pragma: export
#include <vector>

// Test side dual to FUSILLI_CHECK_ERROR. REQUIRE expression that evaluates to
// (or in the case of ErrorOr<T> is convertible to) an ErrorObject to be in ok
// state; exactly equivalent to `REQUIRE(isOk(expr))` but prints a nicer error
// message when a test fails.
//
// Usage:
//   ErrorObject bar();
//
//   TEST_CASE("thing", "[example]") {
//     REQUIRE(isOk(bar()));      // No helpful error message.
//     FUSILLI_REQUIRE_OK(bar()); // Nice error message.
//   }
#define FUSILLI_REQUIRE_OK(expr)                                               \
  do {                                                                         \
    const fusilli::ErrorObject &error = (expr);                                \
    if (isError(error)) {                                                      \
      FUSILLI_LOG_LABEL_RED("ERROR: " << error << " ");                        \
      FUSILLI_LOG_ENDL(#expr << " at " << __FILE__ << ":" << __LINE__);        \
    }                                                                          \
    REQUIRE(isOk(error));                                                      \
  } while (false)

// Unwrap the type returned from an expression that evaluates to an ErrorOr,
// fail the test using Catch2's REQUIRE if the result is an ErrorObject.
//
// This is very similar to FUSILLI_TRY, but FUSILLI_TRY propagates an error to
// callers on the error path, this fails the test on the error path. The two
// macros are analogous to rust's `?` (try) operator and `.unwrap()` call.
#define FUSILLI_REQUIRE_UNWRAP(expr)                                           \
  ({                                                                           \
    auto errorOr = (expr);                                                     \
    FUSILLI_REQUIRE_OK(errorOr);                                               \
    std::move(*errorOr);                                                       \
  })

// Utility to convert vector of dims from int64_t to size_t (unsigned long)
// which is compatible with `iree_hal_dim_t` and fixes narrowing conversion
// warnings.
inline std::vector<size_t> castToSizeT(const std::vector<int64_t> &input) {
  return std::vector<size_t>(input.begin(), input.end());
}

namespace fusilli {

inline ErrorOr<std::shared_ptr<Buffer>>
allocateBufferOfType(Handle &handle, const std::shared_ptr<TensorAttr> &tensor,
                     DataType type, float initVal) {
  FUSILLI_RETURN_ERROR_IF(!tensor, ErrorCode::AttributeNotSet,
                          "Tensor is not set");

  switch (type) {
  case DataType::Float:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
        /*bufferData=*/
        std::vector<float>(tensor->getVolume(), float(initVal)))));
  case DataType::Int32:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
        /*bufferData=*/std::vector<int>(tensor->getVolume(), int(initVal)))));
  case DataType::Half:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
        /*bufferData=*/std::vector<half>(tensor->getVolume(), half(initVal)))));
  case DataType::BFloat16:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
        /*bufferData=*/std::vector<bf16>(tensor->getVolume(), bf16(initVal)))));
  case DataType::Int16:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
        /*bufferData=*/
        std::vector<int16_t>(tensor->getVolume(), int16_t(initVal)))));
  case DataType::Int8:
    return std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
        handle, /*bufferShape=*/castToSizeT(tensor->getPhysicalDim()),
        /*bufferData=*/
        std::vector<int8_t>(tensor->getVolume(), int8_t(initVal)))));
  default:
    return error(ErrorCode::InvalidAttribute, "Unsupported DataType");
  }
}

} // namespace fusilli

#endif // FUSILLI_TESTS_UTILS_H
