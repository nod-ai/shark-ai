# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from shortfin_apps.llm.components.messages import (
    InferencePhase,
    LlmInferenceExecRequest,
)
from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCache,
    BasePagedAttentionCacheAllocation,
)


@pytest.fixture(scope="function")
def mock_base_cache() -> BasePagedAttentionCache:
    return MagicMock(BasePagedAttentionCache)


@patch("shortfin.VoidFuture")
def test_inference_exec_request_repr(mock_void_future):
    """
    Test the string representation of InferenceExecRequest in different states.

    This is useful for debugging and logging. Other test cases may depend on the debug log formats.

    Patches shortfin.VoidFuture with a mock because we're not running this testcase on a worker thread.
    """
    req = LlmInferenceExecRequest(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    instance_id = str(uuid4())
    req.instance_id = instance_id
    assert (
        str(req)
        == f"LlmInferenceExecRequest[phase=P,pos=0,rid=test123,instance_id={instance_id},flags=host,input_token_ids=[1, 2, 3, 4]]"
    )

    req = LlmInferenceExecRequest(InferencePhase.DECODE, [], rid="test123")
    req.return_host_array = False
    req.return_all_logits = False
    req.rid = None
    instance_id = str(uuid4())
    req.instance_id = instance_id
    assert (
        str(req)
        == f"LlmInferenceExecRequest[phase=D,pos=0,rid=None,instance_id={instance_id},flags=,input_token_ids=[]]"
    )


@patch("shortfin.VoidFuture")
def test_copy_exec_request(mock_void_future, mock_base_cache, dummy_pages):
    req = LlmInferenceExecRequest(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    req._cache = mock_base_cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=mock_base_cache)
    req.allocation = allocation
    with patch.object(mock_base_cache, "fork_pages", return_value=allocation):
        new_req = LlmInferenceExecRequest.copy_exec_request(req)
        for attribute in {"start_position", "prompt_length", "_cache"}:
            original_attr = getattr(req, attribute)
            new_attr = getattr(new_req, attribute)
            assert (
                new_attr == original_attr
            ), f"Error copying exec request, expected `{attribute}` to be {original_attr} but got {new_attr}"

        assert (
            new_req.allocation == allocation
        ), f"Error copying exec request, expected `allocation` to be {allocation} but got {new_req.allocation}"


@patch("shortfin.VoidFuture")
def test_inference_exec_request_reset(mock_void_future):
    """
    Test the string representation of InferenceExecRequest in different states.

    This is useful for debugging and logging. Other test cases may depend on the debug log formats.

    Patches shortfin.VoidFuture with a mock because we're not running this testcase on a worker thread.
    """
    req = LlmInferenceExecRequest(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    req.reset(InferencePhase.DECODE)

    assert req.phase == InferencePhase.DECODE
    assert req.return_all_logits == False
    assert req.return_host_array == True
    assert req.result_logits is None


@patch("shortfin.VoidFuture")
def test_cache_page_indices(mock_void_future, mock_base_cache, dummy_pages):
    req = LlmInferenceExecRequest(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    req._cache = mock_base_cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=mock_base_cache)
    req.allocation = allocation

    cache_page_indices = req.cache_page_indices(2)
    assert len(cache_page_indices) == 2


@patch("shortfin.VoidFuture")
def test_publish_allocated_pages(mock_void_future, mock_base_cache, dummy_pages):
    req = LlmInferenceExecRequest(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")

    # Allocation is None
    with pytest.raises(AssertionError):
        req.publish_allocated_pages(len(dummy_pages))

    req._cache = mock_base_cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=mock_base_cache)
    req.allocation = allocation

    req.publish_allocated_pages(len(dummy_pages))


@patch("shortfin.VoidFuture")
def test_free_cache_pages(mock_void_future, mock_base_cache, dummy_pages):
    release_called = False
    req = LlmInferenceExecRequest(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    # Allocation is None
    req.free_cache_pages()
    assert not release_called

    req._cache = mock_base_cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=mock_base_cache)
    req.allocation = allocation
    with patch.object(req.allocation, "release_pages") as mock_release_pages:
        req.free_cache_pages()
        assert req.allocation is None
        mock_release_pages.assert_called_once()
