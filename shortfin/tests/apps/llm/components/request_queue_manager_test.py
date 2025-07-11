# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shortfin.array as sfnp
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.token_selection_strategy.config import DecodeConfig
from shortfin_apps.llm.components.request_queue_manager import RequestQueueManager
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def encoding_4():
    mock_encoding = MagicMock()
    mock_encoding.ids.return_value = [1, 2, 3, 4]
    return mock_encoding


@pytest.fixture
def encoding_2():
    mock_encoding = MagicMock()
    mock_encoding.ids.return_value = [1, 2]
    return mock_encoding


# Use `mock_encoding` in place of `input_batch[0]`


@pytest.fixture
def model_params():
    return ModelParams(
        max_seq_len=512,
        transformer_block_count=42,
        attn_head_dim=42,
        prefill_batch_sizes=[4],
        decode_batch_sizes=[2],
        top_k=5,
        paged_kv_cache=PagedKVCacheParams(
            block_seq_stride=2,
            attention_head_count_kv=42,
            device_block_count=256,
            kv_cache_dtype=sfnp.float16,
        ),
    )


@pytest.fixture
def page_pool():
    return MagicMock()


@pytest.fixture
def responder():
    return MagicMock()


@pytest.fixture
def manager(model_params, page_pool):
    return RequestQueueManager(
        model_params=model_params, page_pool=page_pool, max_queue_size=3
    )


def test_add_to_queue_success(manager, page_pool, responder, encoding_4):
    page_pool.get_available_pages_num.return_value = 100
    decode_config = DecodeConfig(
        num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=10
    )
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding_4],
        is_pretokenized=True,
        responder=responder,
    )
    assert request_id is not None


def test_add_to_queue_full(manager, responder, encoding_2):
    manager._current_queue_size = 3
    decode_config = DecodeConfig(
        num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=10
    )
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding_2],
        is_pretokenized=True,
        responder=responder,
    )
    assert request_id is None
    responder.send_error.assert_called_once()


def test_add_to_queue_topk_mismatch(manager, responder, encoding_2):
    manager.model_params.top_k = 2
    decode_config = DecodeConfig(
        num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=10
    )
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding_2],
        is_pretokenized=True,
        responder=responder,
    )
    assert request_id is None
    responder.send_error.assert_called_once()


def test_add_to_queue_memory_fail(manager, page_pool, responder, encoding_4):
    page_pool.get_available_pages_num.return_value = 1
    decode_config = DecodeConfig(
        num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=100
    )
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding_4],
        is_pretokenized=True,
        responder=responder,
    )
    assert request_id is None
    responder.send_error.assert_called_once()


def test_remove_from_queue_success(manager):
    manager._current_tasks = {1: 1}
    manager._current_queue_size = 1
    manager.remove_from_queue(1)
    assert manager._current_queue_size == 0
    assert 1 not in manager._current_tasks


def test_remove_from_queue_invalid_id(manager):
    with pytest.raises(RuntimeError):
        manager.remove_from_queue(999)


def test_current_tasks(manager):
    manager._current_tasks = {1: 1, 2: 2}
    tasks = manager.current_tasks()
    assert tasks == [1, 2]
