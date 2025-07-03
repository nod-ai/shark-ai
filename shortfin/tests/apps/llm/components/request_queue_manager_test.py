# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from unittest.mock import Mock
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


@pytest.fixture
def mock_model_params():
    mock = Mock()
    mock.paged_kv_cache.block_seq_stride = 32
    return mock


@pytest.fixture
def mock_decode_config():
    mock = Mock()
    mock.num_beams = 8
    mock.max_completion_tokens = 50
    return mock


def test_check_memory_availability_enough_pages(mock_model_params, mock_decode_config):
    limiter = RateLimiter(model_params=mock_model_params)
    input_token_ids_len = 64  # 2 pages
    available_pages = 25  # Should be enough

    result = limiter.check_memory_availability(
        input_token_ids_len=input_token_ids_len,
        available_pages=available_pages,
        decode_config=mock_decode_config,
    )

    assert result is True


def test_check_memory_availability_not_enough_pages(
    mock_model_params, mock_decode_config
):
    limiter = RateLimiter(model_params=mock_model_params)
    input_token_ids_len = 64  # 2 pages
    available_pages = 15  # Not enough

    result = limiter.check_memory_availability(
        input_token_ids_len=input_token_ids_len,
        available_pages=available_pages,
        decode_config=mock_decode_config,
    )

    assert result is False
