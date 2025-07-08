# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shortfin.array as sfnp
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.token_selection_strategy.config import DecodeConfig
from shortfin_apps.llm.components.request_queue_manager import RequestQueueManager

def get_decode_configs(beam_count):
    return [DecodeConfig(eos_token_id=0, num_beams=beam_count)]

def get_model_params(max_queue_size):
    return ModelParams(
        max_seq_len=512,
        transformer_block_count=42,
        attn_head_dim=42,
        prefill_batch_sizes=[4],
        decode_batch_sizes=[max_queue_size],
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

def test_request_queue_manager():
    queue_manager = RequestQueueManager(model_params=get_model_params(6), )

    # Add to queue
    id0 = queue_manager.add_to_queue(get_decode_configs(4))
    assert id0 is not None

    # Try to add beyond max, when `current_queue_size` < `max_queue_size`
    id1 = queue_manager.add_to_queue(get_decode_configs(3))
    assert id1 is None

    # Add more to queue
    id2 = queue_manager.add_to_queue(get_decode_configs(2))
    assert id2 is not None

    # Try to add beyond max
    id3 = queue_manager.add_to_queue(get_decode_configs(2))
    assert id3 is None

    # Remove from queue
    queue_manager.remove_from_queue(id2)

    tasks = queue_manager.current_tasks()
    assert len(tasks) == 1
    assert id0 in tasks

def test_check_memory_availability_enough_pages():
    queue_manager = RequestQueueManager(model_params=get_model_params(4))
    input_token_ids_len = 64  # 2 pages
    available_pages = 25  # Should be enough

    result = queue_manager.check_memory_availability(
        input_token_ids_len=input_token_ids_len,
        available_pages=available_pages,
        decode_config=get_decode_config(8),
    )

    assert result is True


def test_check_memory_availability_not_enough_pages():
    queue_manager = RequestQueueManager(model_params=get_model_params(4))
    input_token_ids_len = 64  # 2 pages
    available_pages = 15  # Not enough

    result = queue_manager.check_memory_availability(
        input_token_ids_len=input_token_ids_len,
        available_pages=available_pages,
        decode_config=get_decode_config(8),
    )

    assert result is False