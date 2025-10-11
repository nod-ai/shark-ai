// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TESTS_HIP_TESTS_HIP_KERNELS_H
#define TESTS_HIP_TESTS_HIP_KERNELS_H

#include <hip/hip_runtime.h>

void launchHelloKernel(dim3 blocks, dim3 threads);

#endif // TESTS_HIP_TESTS_HIP_KERNELS_H
