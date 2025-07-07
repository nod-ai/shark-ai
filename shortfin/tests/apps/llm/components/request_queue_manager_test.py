# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest
import gc

import shortfin.array as sfnp

from shortfin_apps.llm.components.token_selection_strategy.config import DecodeConfig
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.request_queue_manager import RateLimiter
from shortfin_apps.llm.components.request_queue_manager import RequestQueueManager


def test_request_queue_manager():
    queue_manager = RequestQueueManager(6)

    assert queue_manager.current_queue_size == 0

    # Add to queue
    assert queue_manager.add_to_queue(4) == True
    assert queue_manager.current_queue_size == 4

    # Try to add beyond max, when `current_queue_size` < `max_queue_size`
    assert not queue_manager.add_to_queue(3)
    assert queue_manager.current_queue_size == 4

    # Add more to queue
    assert queue_manager.add_to_queue(2) == True
    assert queue_manager.current_queue_size == 6

    # Try to add beyond max
    assert queue_manager.add_to_queue(2) == False
    assert queue_manager.current_queue_size == 6

    # Remove from queue
    queue_manager.remove_from_queue(3)
    assert queue_manager.current_queue_size == 3


def get_model_params():
    return ModelParams(
        max_seq_len=512,
        transformer_block_count=42,
        attn_head_dim=42,
        prefill_batch_sizes=[4],
        decode_batch_sizes=[4],
        paged_kv_cache=PagedKVCacheParams(
            block_seq_stride=32,
            attention_head_count_kv=42,
            device_block_count=256,
            kv_cache_dtype=sfnp.float16,
        ),
    )


def get_decode_config(beam_count):
    return DecodeConfig(
        num_beams=beam_count,
        use_beam_search=True,
        max_completion_tokens=50,
    )


def test_check_memory_availability_enough_pages():
    model_params = get_model_params()
    limiter = RateLimiter(model_params=model_params)
    input_token_ids_len = 64  # 2 pages
    available_pages = 25  # Should be enough

    decode_config = get_decode_config(8)
    result = limiter.check_memory_availability(
        input_token_ids_len=input_token_ids_len,
        available_pages=available_pages,
        decode_config=decode_config,
    )

    assert result is True
    del limiter
    del model_params
    del decode_config
    gc.collect()


def test_check_memory_availability_not_enough_pages():
    model_params = get_model_params()
    limiter = RateLimiter(model_params=model_params)
    input_token_ids_len = 64  # 2 pages
    available_pages = 15  # Not enough

    decode_config = get_decode_config(8)
    result = limiter.check_memory_availability(
        input_token_ids_len=input_token_ids_len,
        available_pages=available_pages,
        decode_config=decode_config,
    )

    assert result is False
    del limiter
    del model_params
    del decode_config
    gc.collect()
