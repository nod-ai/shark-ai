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

#endif // FUSILLI_TESTS_UTILS_H
