# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from sharktank.models.llama.llama import PagedLlamaModelV1
from sharktank.models.llama.toy_llama import generate
from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import QuantizedTensor
from sharktank.utils.create_cache import create_paged_kv_cache
from sharktank.types.theta import flat_to_nested_dict
from sharktank.types.theta import Theta

import pytest
import torch


def test_toyllama_f32():
    theta, config = generate(12345, dtype=torch.float32)
    model = PagedLlamaModelV1(theta=theta, config=config)
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    id_len = len(ids)
    ids = ids + (config.block_seq_stride - len(ids)) * [0]

    ids = torch.asarray([ids], dtype=torch.int64)

    page_ids = torch.asarray([[0]], dtype=torch.int64)
    cache = create_paged_kv_cache(config)
    pages = cache.allocate(128)

    logits = model.prefill(
        tokens=ids, cache_state=pages, attention_mask=None, seq_block_ids=page_ids
    )
    ids = ids[0, 1:id_len]
    logits = logits[0, 0 : (id_len - 1)]
    cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

    assert pytest.approx(0.585, 1e-3) == cross_entropy


def test_toyllama_f16():
    theta, config = generate(12345, dtype=torch.float16)
    model = PagedLlamaModelV1(theta=theta, config=config)
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    id_len = len(ids)
    ids = ids + (config.block_seq_stride - len(ids)) * [0]

    ids = torch.asarray([ids], dtype=torch.int64)

    page_ids = torch.asarray([[0]], dtype=torch.int64)
    cache = create_paged_kv_cache(config)
    pages = cache.allocate(128)

    logits = model.prefill(
        tokens=ids, cache_state=pages, attention_mask=None, seq_block_ids=page_ids
    )
    ids = ids[0, 1:id_len]
    logits = logits[0, 0 : (id_len - 1)]
    cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

    assert pytest.approx(0.585, 1e-3) == cross_entropy


def test_toyllama_f8_as_f32():
    theta, config = generate(12345, dtype=torch.float8_e4m3fnuz)
    config.attention_dtype = torch.float32
    config.activation_dtype = torch.float32

    flat = theta.flatten()
    for k in flat:
        if isinstance(flat[k], QuantizedTensor):
            flat[k] = DefaultPrimitiveTensor(
                name=flat[k].name, data=flat[k].unpack().dequant()
            )
    theta = Theta(flat_to_nested_dict(flat))

    model = PagedLlamaModelV1(theta=theta, config=config)
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    id_len = len(ids)
    ids = ids + (config.block_seq_stride - len(ids)) * [0]

    ids = torch.asarray([ids], dtype=torch.int64)

    page_ids = torch.asarray([[0]], dtype=torch.int64)
    cache = create_paged_kv_cache(config)
    pages = cache.allocate(128)

    logits = model.prefill(
        tokens=ids, cache_state=pages, attention_mask=None, seq_block_ids=page_ids
    )
    ids = ids[0, 1:id_len]
    logits = logits[0, 0 : (id_len - 1)].to(torch.float32)
    cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

    assert pytest.approx(1.357, 1e-3) == cross_entropy


def test_toyllama_f8():
    theta, config = generate(12345, dtype=torch.float8_e4m3fnuz)
    config.attention_dtype = torch.float32
    config.activation_dtype = torch.float32

    model = PagedLlamaModelV1(theta=theta, config=config)
    ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
    id_len = len(ids)
    ids = ids + (config.block_seq_stride - len(ids)) * [0]

    ids = torch.asarray([ids], dtype=torch.int64)

    page_ids = torch.asarray([[0]], dtype=torch.int64)
    cache = create_paged_kv_cache(config)
    pages = cache.allocate(128)

    logits = model.prefill(
        tokens=ids, cache_state=pages, attention_mask=None, seq_block_ids=page_ids
    )
    ids = ids[0, 1:id_len]
    logits = logits[0, 0 : (id_len - 1)].to(torch.float32)
    cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

    assert pytest.approx(1.357, 1e-3) == cross_entropy
