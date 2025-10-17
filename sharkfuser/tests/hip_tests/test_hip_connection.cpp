// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip_utils.h>

#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>

#include <cstdio>

#include "hip_kernels.h"

TEST_CASE("proof of life for HIP", "[hip_tests]") {
  // ----------------------------------------------------------------------
  //  proof of life for GPU connection
  // ----------------------------------------------------------------------

  int dev = 0;
  HIP_REQUIRE_SUCCESS(hipGetDevice(&dev));

  hipDeviceProp_t prop{};
  HIP_REQUIRE_SUCCESS(hipGetDeviceProperties(&prop, dev));

  void *ptr;
  HIP_REQUIRE_SUCCESS(hipMalloc(&ptr, sizeof(float) * 64));

  // Launch kernel (1 block, 4 threads).
  launchHelloKernel(dim3(1), dim3(4));

  HIP_REQUIRE_SUCCESS(hipDeviceSynchronize());

  HIP_REQUIRE_SUCCESS(hipFree(ptr));
}
