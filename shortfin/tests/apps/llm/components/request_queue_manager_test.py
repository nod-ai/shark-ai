# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shortfin.array as sfnp
from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.decode_config import DecodeConfig
from shortfin_apps.llm.components.request_queue_manager import RequestQueueManager
import pytest
from unittest.mock import MagicMock


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
            device_block_count=100,
            kv_cache_dtype=sfnp.float16,
        ),
        has_prefill_position=False,
    )


@pytest.fixture
def responder():
    return MagicMock()


@pytest.fixture
def manager(model_params):
    return RequestQueueManager(
        model_params=model_params,
        max_queue_size=3,
    )


def test_remove_from_queue_success(manager, responder):
    decode_config = DecodeConfig(num_beams=1, top_k=5, max_completion_tokens=10)
    request_id = manager.add_to_queue(
        decode_configs=[decode_config],
        input_batch=[[1, 2]],
        is_pretokenized=True,
        responder=responder,
    )
    used_pages = manager._request_pages[request_id]
    available_before = manager.available_page_count

    manager.remove_from_queue(request_id)

    assert request_id not in manager._current_tasks
    assert request_id not in manager._request_pages
    assert manager.available_page_count == available_before + used_pages


def test_remove_from_queue_invalid_id(manager):
    with pytest.raises(RuntimeError):
        manager.remove_from_queue(999)


def test_current_tasks(manager):
    manager._current_tasks = {1: 1, 2: 2}
    tasks = manager.current_tasks()
    assert tasks == [1, 2]


@pytest.fixture
def mock_model_params():
    return ModelParams(
        max_seq_len=512,
        transformer_block_count=42,
        attn_head_dim=42,
        prefill_batch_sizes=[4],
        decode_batch_sizes=[2],
        top_k=10,
        paged_kv_cache=PagedKVCacheParams(
            block_seq_stride=16,
            attention_head_count_kv=42,
            device_block_count=100,
            kv_cache_dtype=sfnp.float16,
        ),
        has_prefill_position=False,
    )


# Helper function to create mock Encoding objects
def mock_encoding_with_ids(ids_list):
    mock_encoding = MagicMock()
    mock_encoding.ids = ids_list
    return mock_encoding


@pytest.mark.parametrize(
    "decode_configs, input_batch, is_pretokenized, expected_success",
    [
        # One element, is_pretokenized = True
        (
            [
                DecodeConfig(
                    num_beams=1,
                    top_k=5,
                    max_completion_tokens=32,
                )
            ],
            [[1, 2, 3, 4]],
            True,
            True,
        ),
        # Two elements, is_pretokenized = True
        (
            [
                DecodeConfig(
                    num_beams=1,
                    top_k=5,
                    max_completion_tokens=32,
                ),
                DecodeConfig(
                    num_beams=1,
                    top_k=5,
                    max_completion_tokens=32,
                ),
            ],
            [[1, 2], [3, 4]],
            True,
            True,
        ),
        # One element, is_pretokenized = False
        (
            [
                DecodeConfig(
                    num_beams=1,
                    top_k=5,
                    max_completion_tokens=32,
                )
            ],
            [mock_encoding_with_ids([10, 20, 30, 40])],
            False,
            True,
        ),
        # Two elements, is_pretokenized = False
        (
            [
                DecodeConfig(
                    num_beams=1,
                    top_k=5,
                    max_completion_tokens=32,
                ),
                DecodeConfig(
                    num_beams=1,
                    top_k=5,
                    max_completion_tokens=32,
                ),
            ],
            [mock_encoding_with_ids([5, 6, 7]), mock_encoding_with_ids([8, 9])],
            False,
            True,
        ),
        # One element, top_k exceeds exported
        (
            [
                DecodeConfig(
                    num_beams=1,
                    top_k=20,
                    max_completion_tokens=32,
                )
            ],
            [mock_encoding_with_ids([1, 2, 3, 4])],
            False,
            True,
        ),
        # Two elements, memory exceeds
        (
            [
                DecodeConfig(
                    num_beams=10,
                    top_k=5,
                    max_completion_tokens=512,
                ),
                DecodeConfig(
                    num_beams=10,
                    top_k=5,
                    max_completion_tokens=512,
                ),
            ],
            [mock_encoding_with_ids([1] * 100), mock_encoding_with_ids([2] * 100)],
            False,
            False,
        ),
    ],
)
def test_add_to_queue(
    mock_model_params,
    responder,
    decode_configs,
    input_batch,
    is_pretokenized,
    expected_success,
):
    manager = RequestQueueManager(model_params=mock_model_params)
    request_id = manager.add_to_queue(
        decode_configs=decode_configs,
        input_batch=input_batch,
        is_pretokenized=is_pretokenized,
        responder=responder,
    )
    if expected_success:
        assert request_id is not None
        assert request_id in manager.current_tasks()
    else:
        assert request_id is None
