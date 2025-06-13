// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define MAX_K 16 // Upper limit for K, safe for stack on GPU

/*
Batch-enabled TopK Kernel:
- One workgroup per batch (e.g., for input [B, 1, N], grid.x = B)
- Each workgroup processes a single reduction row (1xN)
- Each warp handles the reduction using in-warp TopK logic
*/

extern "C" __global__ void topk_F16I32(const _Float16 *__restrict__ inputValues,
                                       const int32_t *__restrict__ inputIndices,
                                       _Float16 *__restrict__ outputValues,
                                       int32_t *__restrict__ outputIndices,
                                       int reductionSize) {
  int k = 8;
  int groupID = blockIdx.x; // dim 1
  int batchID = blockIdx.y; // dim 0
  int groupCount = gridDim.x;
  uint laneID = threadIdx.x;

  int linearIndex = batchID * groupCount + groupID;
  const _Float16 *batchInput = inputValues + linearIndex * reductionSize;
  const int32_t *batchIndices = inputIndices + linearIndex * reductionSize;
  _Float16 *batchOutputValues = outputValues + linearIndex * k;
  int32_t *batchOutputIndices = outputIndices + linearIndex * k;

  _Float16 NEG_F16_MAX = (_Float16)(-65504.0f);
  _Float16 topk_vals[MAX_K];
  int32_t topk_indices[MAX_K];

  // Initialize topk values to identity (NEG_F16_MAX for max)
  for (int i = 0; i < k; ++i) {
    topk_vals[i] = NEG_F16_MAX;
    topk_indices[i] = -1;
  }

  for (int idx = laneID; idx < reductionSize; idx += groupCount) {
    _Float16 val = batchInput[idx];
    int32_t ind = batchIndices[idx];

    // Insert into local top-k buffer
    for (int j = 0; j < k; ++j) {
      if (val > topk_vals[j]) {
        _Float16 tmp_val = topk_vals[j];
        int32_t tmp_ind = topk_indices[j];

        topk_vals[j] = val;
        topk_indices[j] = ind;

        val = tmp_val;
        ind = tmp_ind;
      }
    }
  }

  // Collect and merge top-k from all lanes
  __shared__ _Float16 warp_topk_vals[128 * MAX_K];
  __shared__ int32_t warp_topk_indices[128 * MAX_K];

  for (int i = 0; i < k; ++i) {
    warp_topk_vals[laneID * k + i] = topk_vals[i];
    warp_topk_indices[laneID * k + i] = topk_indices[i];
  }

  __syncthreads();

  // Merge in lane 0
  if (laneID == 0) {
    // Naive partial sort of k * groupCount
    for (int i = k; i < groupCount * k; ++i) {
      _Float16 hold_v = warp_topk_vals[i];
      int32_t hold_i = warp_topk_indices[i];

      for (int j = 0; j < k; ++j) {
        if (warp_topk_vals[j] < hold_v) {
          _Float16 tmp_v = warp_topk_vals[j];
          int32_t tmp_i = warp_topk_indices[j];
          warp_topk_vals[j] = hold_v;
          warp_topk_indices[j] = hold_i;
          hold_v = tmp_v;
          hold_i = tmp_i;
        }
      }
    }
    for (int i = 0; i < k; ++i) {
      batchOutputValues[i] = warp_topk_vals[i];
      batchOutputIndices[i] = warp_topk_indices[i];
    }
  }
}
