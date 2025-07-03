# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import unittest

import torch

from itertools import product
from parameterized import parameterized

from sharktank.models.llm import *
from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.utils.export_artifacts import IreeCompileException
from sharktank.utils.load_llm import *
from sharktank.utils.evaluate import *
from sharktank.utils.testing import (
    is_mi300x,
    IreeVsEagerLLMTester,
    TempDirTestBase,
    xfail,
)


class DeepseekCrossEntropyTest(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float16, torch.float32),
            (torch.float32, torch.float32),
        ]
    )
    def testUnsharded(self, dtype_rest: torch.dtype, dtype_norm: torch.dtype):
        theta, config = generate(12345, dtype_rest=dtype_rest, dtype_norm=dtype_norm)
        model = PagedLlmModelV1(theta=theta, config=config)

        ids = [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]]

        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)

        generator = TorchGenerator(model)
        batch = generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )

        batch.prefill()
        logits = batch.prefill_logits

        ids = token_ids[0, :-1]
        logits = logits[0, 1:]
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

        assert pytest.approx(9.7477, 1e-4) == cross_entropy


@pytest.mark.usefixtures("iree_flags", "device")
@is_mi300x
class DeepseekIreeVsEagerTest(TempDirTestBase):
    @parameterized.expand(product([1, 2], [1, 2]))
    @xfail(
        raises=IreeCompileException,
        reason="https://github.com/iree-org/iree/issues/21165",
        strict=True,
        match="op write affecting operations on global resources are restricted to workgroup",
    )
    def testUnshardedToyIreeVsEager(
        self, tensor_parallelism_size: int, pipeline_parallelism_size: int
    ):
        theta, config = generate(12345)
        config.tensor_parallelism_size = tensor_parallelism_size
        config.pipeline_parallelism_size = pipeline_parallelism_size

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
        except IreeCompileException as e:
            if tensor_parallelism_size == 2:
                pytest.xfail(reason="https://github.com/iree-org/iree/issues/20354")
            else:
                raise e
        tester.run_and_compare_iree_vs_eager()
