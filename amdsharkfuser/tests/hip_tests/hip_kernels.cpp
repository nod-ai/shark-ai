// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

__global__ void helloKernel() {
  printf("Hello from GPU! block %d thread %d\n", blockIdx.x, threadIdx.x);
}

void launchHelloKernel(dim3 blocks, dim3 threads) {
  hipLaunchKernelGGL(helloKernel, blocks, threads, 0, 0);
}
