# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from sharktank.models.deepseek.deepseek import PagedDeepseekModelV1
from sharktank.models.deepseek.toy_deepseek import generate

import pytest
import torch


def test_deepseek():
    torch.set_default_dtype(torch.float32)
    theta, config = generate(12345)
    model = PagedDeepseekModelV1(theta=theta, config=config)

    ids = torch.asarray(
        [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]], dtype=torch.int64
    )
    logits = model.prefill(tokens=ids)
    ids = ids[0, :-1]
    logits = logits[0, 1:]
    cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

    assert pytest.approx(5.3062, 1e-4) == cross_entropy
