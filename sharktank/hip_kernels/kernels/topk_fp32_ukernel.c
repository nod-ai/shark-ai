// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define MAX_K 8 // Upper limit for K, safe for stack on GPU

/*
Batch-enabled TopK Kernel:
- One workgroup per batch (e.g., for input [B, 1, N], grid.x = B)
- Each workgroup processes a single reduction row (1xN)
- Each warp handles the reduction using in-warp TopK logic
*/

extern "C" __global__ void topk_F32I32(const float *__restrict__ inputValues,
                                       const int32_t *__restrict__ inputIndices,
                                       float *__restrict__ outputValues,
                                       int32_t *__restrict__ outputIndices,
                                       int reductionSize) {
  int k = 8;
  int groupID = blockIdx.x; // dim 1
  int threadCount = blockDim.x;
  uint laneID = threadIdx.x;

  int linearIndex = groupID;
  int64_t linearIndex64 = linearIndex;
  int64_t reductionSize64 = reductionSize;
  int64_t reductionOffset = linearIndex64 * reductionSize64;

  const float *batchInput = inputValues + reductionOffset;
  const int32_t *batchIndices = inputIndices + reductionOffset;
  float *batchOutputValues = outputValues + linearIndex * k;
  int32_t *batchOutputIndices = outputIndices + linearIndex * k;

  float NEG_F16_MAX = (float)(-3.4028234663852885981170418348451692544e+38);

  int warp_offset = laneID * k;

  // Collect and merge top-k from all lanes
  __shared__ float warp_topk_vals[64 * MAX_K];
  __shared__ int32_t warp_topk_indices[64 * MAX_K];

  // Initialize topk values to identity (NEG_F16_MAX for max)
  for (int i = 0; i < k; ++i) {
    warp_topk_vals[warp_offset + i] = NEG_F16_MAX;
    warp_topk_indices[warp_offset + i] = -1;
  }

  for (int i = laneID; i < reductionSize; i += threadCount) {
    float val = batchInput[i];
    int32_t ind = batchIndices[i];

    // Insert into local top-k buffer
    for (int j = 0; j < k; ++j) {
      int idx = warp_offset + j;
      float curr_val = warp_topk_vals[idx];
      int32_t curr_ind = warp_topk_indices[idx];
      bool swap = val > curr_val;
      float new_val = swap ? val : curr_val;
      int32_t new_ind = swap ? ind : curr_ind;
      val = swap ? curr_val : val;
      ind = swap ? curr_ind : ind;
      warp_topk_vals[idx] = new_val;
      warp_topk_indices[idx] = new_ind;
    }
  }

  __syncthreads();

  int SUBGROUPS = 16;

  // Merge in lane 0
  if (laneID < SUBGROUPS) {
    // Naive partial sort of k * warpSize
    for (int i = laneID + SUBGROUPS * k; i < threadCount * k; i += SUBGROUPS) {
      float val = warp_topk_vals[i];
      int32_t ind = warp_topk_indices[i];

      // Insert into local top-k buffer
      for (int j = 0; j < k; ++j) {
        int idx = warp_offset + j;
        float curr_val = warp_topk_vals[idx];
        int32_t curr_ind = warp_topk_indices[idx];
        bool swap = val > curr_val;
        float new_val = swap ? val : curr_val;
        int32_t new_ind = swap ? ind : curr_ind;
        val = swap ? curr_val : val;
        ind = swap ? curr_ind : ind;
        warp_topk_vals[idx] = new_val;
        warp_topk_indices[idx] = new_ind;
      }
    }
  }

  __syncthreads();

  // Merge in lane 0
  if (laneID == 0) {
    // Naive partial sort of k * warpSize
    for (int i = k; i < SUBGROUPS * k; ++i) {
      float val = warp_topk_vals[i];
      int32_t ind = warp_topk_indices[i];

      // Insert into local top-k buffer
      for (int j = 0; j < k; ++j) {
        int idx = j;
        float curr_val = warp_topk_vals[idx];
        int32_t curr_ind = warp_topk_indices[idx];
        bool swap = val > curr_val;
        float new_val = swap ? val : curr_val;
        int32_t new_ind = swap ? ind : curr_ind;
        val = swap ? curr_val : val;
        ind = swap ? curr_ind : ind;
        warp_topk_vals[idx] = new_val;
        warp_topk_indices[idx] = new_ind;
      }
    }
    for (int i = 0; i < k; ++i) {
      batchOutputValues[i] = warp_topk_vals[i];
      batchOutputIndices[i] = warp_topk_indices[i];
    }
  }
}
