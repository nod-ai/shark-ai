// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusili;

TEST_CASE("Convolution fprop", "[conv][graph]") {

  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  // Placeholder for the proper conv sample, just to get dir structure in place
  REQUIRE(true);
}
