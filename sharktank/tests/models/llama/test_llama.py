# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from sharktank.models.llama.llama import PagedLlamaModelV1
from sharktank.models.llama.toy_llama import generate

import pytest
import torch


def test_llama():
    torch.set_default_dtype(torch.float32)
    theta, config = generate(12345)
    model = PagedLlamaModelV1(theta=theta, config=config)

    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
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
        attention_mask=None,
        cache_state=cache_state,
        seq_block_ids=block_ids,
    )

    # Remove padding
    ids = ids[:, :seq_len]
    logits = logits[:, :seq_len, :]

    ids = ids[0, 1:]
    logits = logits[0, :-1].to(torch.float32)
    cross_entropy = torch.nn.functional.cross_entropy(logits, ids)
    assert pytest.approx(0.577, 1e-2) == cross_entropy
