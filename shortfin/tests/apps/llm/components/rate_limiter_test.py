# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace


@pytest.fixture
def mock_model_params():
    mock = MagicMock()
    mock.paged_kv_cache.block_seq_stride = 4
    return mock

@pytest.mark.parametrize("num_beams,input_len,available_pages,expected", [
    (8, 16, 12, True),   # ceil(16/4) + 8 - 1 = 11
    (8, 16, 10, False),  # 11 > 10
    (1, 12, 4, True),    # ceil(12/4) + 1 - 1 = 3
    (1, 12, 2, False),   # 3 > 2
])
def test_check_memory_availability(mock_model_params, num_beams, input_len, available_pages, expected):
    mock_server_params = MagicMock()
    mock_server_params.decode_config = SimpleNamespace(num_beams=num_beams)

    rate_limiter = RateLimiter(
        model_params=mock_model_params,
        server_params=mock_server_params
    )

    assert rate_limiter.check_memory_availability(
        input_token_ids_len=input_len,
        available_pages=available_pages
    ) == expected
