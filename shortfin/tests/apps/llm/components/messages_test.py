# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)
from unittest.mock import patch


@patch("shortfin.VoidFuture")
def test_inference_exec_request_repr(mock_void_future):
    """
    Test the string representation of InferenceExecRequest in different states.

    This is useful for debugging and logging. Other test cases may depend on the debug log formats.

    Patches shortfin.VoidFuture with a mock because we're not running this testcase on a worker thread.
    """
    req = LlmInferenceExecRequest(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    assert (
        str(req)
        == "LlmInferenceExecRequest[phase=P,pos=0,rid=test123,flags=host,input_token_ids=[1, 2, 3, 4]]"
    )

    req = LlmInferenceExecRequest(InferencePhase.DECODE, [], rid="test123")
    req.return_host_array = False
    req.return_all_logits = False
    req.rid = None
    assert (
        str(req)
        == "LlmInferenceExecRequest[phase=D,pos=0,rid=None,flags=,input_token_ids=[]]"
    )
