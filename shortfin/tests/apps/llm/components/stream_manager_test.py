# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin_apps.llm.components.stream_manager import StreamManager
from shortfin_apps.llm.components.manager import LlmSystemManager

import pytest


@pytest.fixture
def disagg_sysman() -> LlmSystemManager:
    sysman = LlmSystemManager(
        device="amdgpu", device_ids="0", disaggregated_invocation=True
    )
    return sysman


@pytest.fixture
def non_disagg_sysman() -> LlmSystemManager:
    sysman = LlmSystemManager(device="amdgpu")
    return sysman


@pytest.fixture
def disagg_stream_manager(disagg_sysman) -> StreamManager:
    sman = StreamManager(disagg_sysman, 32)
    return sman


@pytest.fixture
def non_disagg_stream_manager(non_disagg_sysman) -> StreamManager:
    sman = StreamManager(non_disagg_sysman, 32)
    return sman


def test_stream_manager_disagg_is_two_streams(disagg_stream_manager):
    assert disagg_stream_manager.num_open_streams() == 2


def test_stream_manager_non_disagg_is_one_stream(non_disagg_stream_manager):
    assert non_disagg_stream_manager.num_open_streams() == 1


def test_stream_manager_disagg_returns_alternate_streams(disagg_stream_manager):
    assert disagg_stream_manager.num_open_streams() == 2

    num_stream_requests = 20
    cached_device = None
    for i in range(num_stream_requests):
        idx, device = disagg_stream_manager.get_stream()
        assert idx == i % 2
        assert device != cached_device
        cached_device = device


def test_stream_manager_non_disagg_returns_same_stream(non_disagg_stream_manager):
    assert non_disagg_stream_manager.num_open_streams() == 1

    num_stream_requests = 20
    cached_device = None
    for i in range(num_stream_requests):
        idx, device = non_disagg_stream_manager.get_stream()
        assert idx == 0
        if i > 0:
            assert device == cached_device
        cached_device = device
