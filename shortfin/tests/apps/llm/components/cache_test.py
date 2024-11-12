# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for llm kvcache component.
"""

import pytest
import time
import tempfile
import shortfin as sf
from _shortfin import lib as sfl
from shortfin_apps.llm.components import cache
from shortfin_apps.llm.components import config_struct
import json
from pathlib import Path


@pytest.fixture
def lsys():
    sc = sfl.local.host.CPUSystemBuilder()
    ls = sc.create_system()
    yield ls
    ls.shutdown()


@pytest.fixture
def fiber(lsys):
    # TODO: Should adopt the main thread.
    worker = lsys.create_worker("main")
    return lsys.create_fiber(worker)


@pytest.fixture
def device(fiber):
    return fiber.device(0)


@pytest.fixture
def model_params():
    model_params = {
        "module_name": "module",
        "module_abi_version": 1,
        "max_seq_len": 2048,
        "attn_head_dim": 100,
        "attn_head_count": 32,
        "prefill_batch_sizes": [4],
        "decode_batch_sizes": [4],
        "transformer_block_count": 26,
        "paged_kv_cache": {
            "block_seq_stride": 16,
            "device_block_count": 256,
            "attn_head_count_kv": 32,
        },
    }

    # Create a temporary file to store the JSON
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        json.dump(model_params, tmp_file, indent=4)
        tmp_path = Path(tmp_file.name)

    try:
        # Load the JSON using config_struct
        model_params = config_struct.ModelParams.load_json(tmp_path)
        yield model_params
    finally:
        tmp_path.unlink


@pytest.fixture
def cache_fixture(fiber, model_params) -> cache.AttnPageCache:
    # Create and return the cache object
    return cache.AttnPageCache(
        devices=fiber.devices_dict.values(), model_params=model_params
    )


@pytest.mark.parametrize("n_allocated", [1, 16, 255])
def test_alloc(
    cache_fixture: cache.AttnPageCache,
    n_allocated,
    model_params: config_struct.ModelParams,
):
    alloc_page_count = cache_fixture.page_tables[0].shape[0]

    assert alloc_page_count == model_params.paged_kv_cache.device_block_count

    pages = cache_fixture.acquire_free_pages(n_allocated)
    last_page = alloc_page_count - 1
    expected_indices = range(last_page, last_page - n_allocated, -1)
    for p, expected_ix in zip(pages, expected_indices):
        assert p.index == expected_ix
        assert p.index > 0
