# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from shortfin_apps.llm.components.messages import InferenceExecRequest, InferencePhase


@pytest.fixture
def mock_request():
    """
    A mock InferenceExecRequest class that doesn't require worker context.

    Useful for testing init and printing.
    """
    original_init = InferenceExecRequest.__init__

    def mock_init(self, phase, input_token_ids, rid=None):
        self.phase = phase
        self.start_position = 0
        self.input_token_ids = input_token_ids
        self.rid = rid
        self.return_all_logits = False
        self.return_host_array = True
        self.result_logits = None
        self._cache = None
        self.allocation = None

    InferenceExecRequest.__init__ = mock_init
    yield InferenceExecRequest
    InferenceExecRequest.__init__ = original_init


def test_inference_exec_request_repr(mock_request):
    """
    Test the string representation of InferenceExecRequest in different states.

    This is useful for debugging and logging. Other test cases may depend on the debug log formats.
    """
    # default settings
    req = mock_request(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    assert (
        str(req)
        == "InferenceExecRequest[phase=P,pos=0,rid=test123,flags=host,input_token_ids=[1, 2, 3, 4]]"
    )

    # mutated settings
    req = mock_request(InferencePhase.DECODE, [], rid="test123")
    req.return_host_array = False
    req.return_all_logits = False
    req.rid = None
    assert (
        str(req)
        == "InferenceExecRequest[phase=D,pos=0,rid=None,flags=,input_token_ids=[]]"
    )
