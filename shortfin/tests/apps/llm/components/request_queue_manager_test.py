# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shortfin.array as sfnp
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.token_selection_strategy.config import DecodeConfig
from shortfin_apps.llm.components.request_queue_manager import RequestQueueManager
from shortfin_apps.llm.components.tokenizer import Encoding


@pytest.fixture
def model_params():
    return ModelParams(
        decode_batch_sizes=[2],
        top_k=5,
        paged_kv_cache=PagedKVCacheParams(block_seq_stride=2)
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
        model_params=model_params,
        page_pool=page_pool,
        max_queue_size=3
    )

def test_add_to_queue_success(manager, page_pool, responder):
    page_pool.get_available_pages_num.return_value = 100
    decode_config = DecodeConfig(num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=10)
    encoding = Encoding(ids=[1, 2, 3, 4])
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding],
        is_pretokenized=True,
        responder=responder
    )
    assert request_id is not None

def test_add_to_queue_full(manager, responder):
    manager._current_queue_size = 3
    decode_config = DecodeConfig(num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=10)
    encoding = Encoding(ids=[1, 2])
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding],
        is_pretokenized=True,
        responder=responder
    )
    assert request_id is None
    responder.send_error.assert_called_once()

def test_add_to_queue_topk_mismatch(manager, responder):
    manager.model_params.top_k = 2
    decode_config = DecodeConfig(num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=10)
    encoding = Encoding(ids=[1, 2])
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding],
        is_pretokenized=True,
        responder=responder
    )
    assert request_id is None
    responder.send_error.assert_called_once()

def test_add_to_queue_memory_fail(manager, page_pool, responder):
    page_pool.get_available_pages_num.return_value = 1
    decode_config = DecodeConfig(num_beams=1, top_k=5, use_beam_search=False, max_completion_tokens=100)
    encoding = Encoding(ids=[1, 2, 3, 4])
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[encoding],
        is_pretokenized=True,
        responder=responder
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
