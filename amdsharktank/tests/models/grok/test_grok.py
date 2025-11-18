# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktank.models.llm import *
from amdsharktank.models.grok.toy_grok import generate

import torch
import pytest


@pytest.mark.xfail(
    raises=AssertionError,
    reason="https://github.com/nod-ai/amdshark-ai/issues/1270",
)
def test_grok():
    theta, config = generate(12345)
    model = PagedLlmModelV1(theta=theta, config=config)

    ids = [0, 102, 133, 192, 153, 26, 172, 3, 41, 193, 78, 204, 38, 30, 11, 62, 192, 38]
    seq_len = len(ids)

    blocks = (seq_len - 1) // config.block_seq_stride
    blocks = blocks + 1
    padded_length = blocks * config.block_seq_stride
    padding = padded_length - seq_len
    ids = ids + [0] * padding

    ids = torch.asarray([ids], dtype=torch.int64)
    block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

    cache_state = model.cache.allocate(
        page_count=config.hp.context_length // config.block_seq_stride
    )

    logits = model.prefill(
        tokens=ids,
        seq_lens=torch.tensor([seq_len]),
        cache_state=cache_state,
        seq_block_ids=block_ids,
    )

    # Remove padding
    ids = ids[:, :seq_len]
    logits = logits[:, :seq_len, :]

    ids = ids[0, 1:].cpu()
    logits = logits[0, :-1].to(torch.float32).cpu()
    cross_entropy = torch.nn.functional.cross_entropy(logits, ids)
    assert pytest.approx(2.0267, 1e-2) == cross_entropy
