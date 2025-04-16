# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import random

import shortfin.array as sfnp

from shortfin_apps.llm.components.token_selection_strategy.sampler import Sampler
from shortfin_apps.utils import convert_int_to_float, convert_float_to_int


def test_sampler_select_top_k(device):
    sampler = Sampler()

    # Sorted ascending
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data
    k = 8
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert top_tokens == [i for i in range(8, 16)]
    assert top_values == [i for i in range(8, 16)]

    # Sorted descending
    data = data[::-1]
    src.items = data
    k = 8
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert sorted(top_tokens) == [i for i in range(0, 8)]
    assert sorted(top_values) == [i for i in range(8, 16)]

    # Randomized data
    random.shuffle(data)
    src.items = data
    k = 5
    expected_values = {val for val in range(11, 16)}
    expected_tokens = [i for i in range(len(data)) if data[i] in expected_values]
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert sorted(top_tokens) == expected_tokens
    assert sorted(top_values) == list(expected_values)


def test_sampler_select_top_k_one_dim(device):
    sampler = Sampler()

    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)

    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data
    k = 8
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert top_tokens == [i for i in range(8, 16)]
    assert top_values == [i for i in range(8, 16)]

    # Sorted descending
    data = data[::-1]
    src.items = data
    k = 8
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert sorted(top_tokens) == [i for i in range(0, 8)]
    assert sorted(top_values) == [i for i in range(8, 16)]

    # Randomized data
    random.shuffle(data)
    src.items = data
    k = 5
    expected_values = {val for val in range(11, 16)}
    expected_tokens = [i for i in range(len(data)) if data[i] in expected_values]
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert sorted(top_tokens) == expected_tokens
    assert sorted(top_values) == list(expected_values)


def test_sampler_select_top_k_float16(device):
    sampler = Sampler()

    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float16)
    data = [
        convert_float_to_int(float(i), sfnp.float16)
        for i in range(math.prod(src.shape))
    ]
    src.items = data
    k = 8
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert top_tokens == [i for i in range(8, 16)]
    assert top_values == [i for i in range(8, 16)]

    # Randomize data
    random.shuffle(data)
    src.items = data
    k = 5
    expected_values = {val for val in range(11, 16)}
    expected_tokens = [
        i
        for i in range(len(data))
        if convert_int_to_float(data[i], sfnp.float16) in expected_values
    ]
    top_tokens, top_values = sampler.select_top_k(src, -k)
    assert sorted(top_tokens) == expected_tokens
    assert sorted(top_values) == list(expected_values)


def test_sampler_sample_top_k(device):
    sampler = Sampler()

    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [float(i) for i in range(math.prod(src.shape))]
    src.items = data

    # One hot
    tokens = [i for i in range(11)]
    probs = [0.0 for _ in range(len(tokens))]

    hot_token = random.randint(0, 10)
    probs[hot_token] = 1.0

    expected_tokens = {hot_token}
    expected_probs = {1.0}

    for k in range(1, 10):
        result_tokens, result_probs = sampler.sample_top_k(tokens, probs, k)
        assert len(result_tokens) == k
        assert len(result_probs) == k
        assert all(token in expected_tokens for token in result_tokens)
        assert all(prob in expected_probs for prob in result_probs)

    # Two hot
    second_hot_token = hot_token
    while second_hot_token == hot_token:
        second_hot_token = random.randint(0, 10)

    expected_tokens = {hot_token, second_hot_token}
    expected_probs = {0.75, 0.25}

    probs[hot_token] = 0.75
    probs[second_hot_token] = 0.25

    for k in range(1, 10):
        result_tokens, result_probs = sampler.sample_top_k(tokens, probs, k)
        assert len(result_tokens) == k
        assert len(result_probs) == k
        assert all(token in expected_tokens for token in result_tokens)
        assert all(prob in expected_probs for prob in result_probs)
