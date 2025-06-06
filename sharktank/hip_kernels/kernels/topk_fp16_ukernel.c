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

extern "C" __global__ void topk_F16I32(const _Float16 *__restrict__ inputBuffer,
                                       _Float16 *__restrict__ outputValues,
                                       int32_t *__restrict__ outputIndices,
                                       int reductionSize) {
  int k = 8;
  int groupID = blockIdx.x; // dim 1
  int batchID = blockIdx.y; // dim 0
  int groupCount = gridDim.x;
  uint laneID = threadIdx.x;

  int linearIndex = batchID * groupCount + groupID;
  const _Float16 *batchInput = inputBuffer + linearIndex * reductionSize;
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

  uint numBatches = (reductionSize + warpSize - 1) / warpSize;
  for (int i = 0; i < numBatches; ++i) {
    uint idx = warpSize * i + laneID;
    _Float16 val = idx < reductionSize ? batchInput[idx] : NEG_F16_MAX;

    // Insert into local top-k buffer
    for (int j = 0; j < k; ++j) {
      if (val > topk_vals[j]) {
        // Shift down
        for (int m = k - 1; m > j; --m) {
          topk_vals[m] = topk_vals[m - 1];
          topk_indices[m] = topk_indices[m - 1];
        }
        topk_vals[j] = val;
        topk_indices[j] = idx;
        break;
      }
    }
  }

  // Collect and merge top-k from all lanes
  __shared__ _Float16 warp_topk_vals[warpSize * MAX_K];
  __shared__ int32_t warp_topk_indices[warpSize * MAX_K];

  for (int i = 0; i < k; ++i) {
    warp_topk_vals[laneID * k + i] = topk_vals[i];
    warp_topk_indices[laneID * k + i] = topk_indices[i];
  }

  __syncthreads();

  int SUBGROUPS = 16;

  if (laneID < SUBGROUPS) {
    // Naive partial sort of k * warpSize
    for (int i = laneID + k * SUBGROUPS; i < warpSize * k; i += SUBGROUPS) {
      _Float16 hold_v = warp_topk_vals[i];
      int32_t hold_i = warp_topk_indices[i];

      for (int j = 0; j < k; ++j) {
        int IDX = j + laneID * k;
        if (warp_topk_vals[IDX] < hold_v) {
          _Float16 tmp_v = warp_topk_vals[IDX];
          int32_t tmp_i = warp_topk_indices[IDX];
          warp_topk_vals[IDX] = hold_v;
          warp_topk_indices[IDX] = hold_i;
          hold_v = tmp_v;
          hold_i = tmp_i;
        }
      }
    }
  }

  __syncthreads();

  // Merge in lane 0
  if (laneID == 0) {
    // Naive partial sort of k * warpSize
    for (int i = k; i < SUBGROUPS * k; ++i) {
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
