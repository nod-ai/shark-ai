# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import pytest
from unittest.mock import MagicMock
from shortfin_apps.llm.components.rate_limiter import RateLimiter


def create_rate_limiter(stride: int, num_beams: int):
    mock_model_params = MagicMock()
    mock_model_params.paged_kv_cache.block_seq_stride = stride

    mock_server_params = MagicMock()
    mock_server_params.decode_config.num_beams = num_beams

    return RateLimiter(model_params=mock_model_params, server_params=mock_server_params)


@pytest.mark.parametrize(
    "num_beams, input_len, available_pages, expected",
    [
        (1, 64, 2, True),  # needed_pages = ceil(64/32) + 1 - 1 = 2
        (1, 64, 1, False),  # needed_pages = 2
        (8, 64, 9, True),  # needed_pages = ceil(64/32) + 8 - 1 = 9
        (8, 64, 8, False),  # needed_pages = 9
    ],
)
def test_check_memory_availability(num_beams, input_len, available_pages, expected):
    rate_limiter = create_rate_limiter(stride=32, num_beams=num_beams)
    assert (
        rate_limiter.check_memory_availability(
            input_token_ids_len=input_len, available_pages=available_pages
        )
        is expected
    )
