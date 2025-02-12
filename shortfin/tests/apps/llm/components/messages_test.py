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
    # Basic request with default settings
    req = mock_request(InferencePhase.PREFILL, [1, 2, 3, 4], rid="test123")
    assert (
        str(req)
        == "InferenceExecRequest[phase=P,pos=0,rid=test123,flags=host,input_token_ids=[1, 2, 3, 4]]"
    )

    # With mutated flags
    req.return_all_logits = True
    req.phase = InferencePhase.DECODE
    req.rid = 1
    assert (
        str(req)
        == "InferenceExecRequest[phase=D,pos=0,rid=1,flags=all,host,input_token_ids=[1, 2, 3, 4]]"
    )

    # With no flags set
    req.return_host_array = False
    req.return_all_logits = False
    req.rid = None
    req.input_token_ids = list(range(100))
    assert (
        str(req)
        == "InferenceExecRequest[phase=P,pos=0,rid=test123,flags=,input_token_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]"
    )
