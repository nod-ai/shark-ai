# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import unittest

import pytest
import torch

from itertools import product
from parameterized import parameterized

from sharktank.models.llm import *
from sharktank.models.llama.toy_llama import generate
from sharktank.utils.export_artifacts import IreeCompileException
from sharktank.utils.testing import (
    is_mi300x,
    IreeVsEagerLLMTester,
    TempDirTestBase,
)


class CrossEntropyTest(unittest.TestCase):
    def testUnsharded(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
        seq_len = len(ids)

        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        ids = ids + [0] * padding

        ids = torch.asarray([ids], dtype=torch.int64)
        block_ids = [torch.asarray([[i for i in range(blocks)]]).to(torch.int64)]

        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        logits = model.prefill(
            tokens=ids,
            attention_mask=[None],
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding
        ids = ids[:, :seq_len]
        logits = logits[:, :seq_len, :]

        ids = ids[0, 1:]
        logits = logits[0, :-1].to(torch.float32)
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)
        assert pytest.approx(0.583, 1e-2) == cross_entropy


@pytest.mark.usefixtures("iree_flags", "device")
@is_mi300x
class LlamaIreeVsEagerTest(TempDirTestBase):
    @parameterized.expand(product([1, 2], [1, 2], [False, True]))
    @pytest.mark.xfail(
        raises=AssertionError,
        reason="https://github.com/nod-ai/shark-ai/issues/1758",
        strict=True,
    )
    def testUnshardedToyIreeVsEager(
        self, tensor_parallelism_size: int, pipeline_parallelism_size: int, use_hf: bool
    ):
        theta, config = generate(12345)
        config.tensor_parallelism_size = tensor_parallelism_size
        config.pipeline_parallelism_size = pipeline_parallelism_size
        config.use_hf = use_hf

        try:
            tester = IreeVsEagerLLMTester(
                work_dir=self._temp_dir,
                theta=theta,
                config=config,
                torch_device=self.device,
                iree_device=self.iree_device,
                iree_hip_target=self.iree_hip_target,
                iree_hal_target_device=self.iree_hal_target_device,
            )
            tester.run_and_compare_iree_vs_eager()
        except IreeCompileException as e:
            if tensor_parallelism_size == 2 or pipeline_parallelism_size == 2:
                pytest.xfail(reason="sharding compilation is broken")
            else:
                raise e
