# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *

import torch

__all__ = [
    "paged_attention_kv_cache_gather",
]

CACHE_SIZE = DynDim.CACHE_SIZE
PAGES = DynDim.PAGES
T_BLOCK = StaticDim.T_BLOCK
PART = StaticDim.PART
BLOCK_SEQ_STRIDE = StaticDim.BLOCK_SEQ_STRIDE
HEAD_COUNT_KV = StaticDim.HEAD_COUNT_KV
ATTN_HEAD_DIM = StaticDim.ATTN_HEAD_DIM
BATCH = DynDim.BATCH

CACHE_TY = Dtype.CACHE_TY
I64 = Dtype.I64


@mlir_kernel(
    inputs=(
        MLIRTensor[
            CACHE_SIZE,
            T_BLOCK,
            PART,
            HEAD_COUNT_KV,
            BLOCK_SEQ_STRIDE,
            ATTN_HEAD_DIM,
            CACHE_TY,
        ],
        MLIRTensor[BATCH, PAGES, I64],
        MLIRTensor[I64],
        MLIRTensor[I64],
    ),
    results=(
        MLIRTensor[
            BATCH, PAGES, HEAD_COUNT_KV, BLOCK_SEQ_STRIDE, ATTN_HEAD_DIM, CACHE_TY
        ],
    ),
)
def paged_attention_kv_cache_gather(
    cache, page_ids, transformer_idx, partition_idx, result=None
):
    mlir = """
    !cache_slice = tensor<{{[CACHE_SIZE, HEAD_COUNT_KV, BLOCK_SEQ_STRIDE, ATTN_HEAD_DIM]|join('x')}}x!cache_dtype>

    module {

    util.func private @{{kernel_name}}(
        %cache: !cache, %page_ids: !page_ids, %transformer_idx: !transformer_idx, %partition_idx: !partition_idx) -> !result {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Get transformer/partition ids.
    %t_id64 = tensor.extract %transformer_idx[] : !transformer_idx
    %p_id64 = tensor.extract %partition_idx[] : !partition_idx
    %t_id = arith.index_cast %t_id64 : !transformer_idx_dtype to index
    %p_id = arith.index_cast %p_id64 : !partition_idx_dtype to index

    // Get dynamic dimensions.
    %cache_size = tensor.dim %cache, %c0 : !cache
    %batches = tensor.dim %page_ids, %c0 : !page_ids
    %pages = tensor.dim %page_ids, %c1 : !page_ids

    // Extract a the current transformer block and partition from cache.
    %cache_slice = tensor.extract_slice %cache [0, %t_id, %p_id, 0, 0, 0] [%cache_size, 1, 1, {{HEAD_COUNT_KV}}, {{BLOCK_SEQ_STRIDE}}, {{ATTN_HEAD_DIM}}] [1, 1, 1, 1, 1, 1] : !cache to !cache_slice

    %empty = tensor.empty(%batches, %pages) : !result

    // Gather from cache_slice using page_ids.
    %result = iree_linalg_ext.gather
        dimension_map = [0]
        ins(%cache_slice, %page_ids : !cache_slice, !page_ids)
        outs(%empty : !result) -> !result

    util.return %result : !result
    }
    }
    """
    return MLIRSpec(mlir)
