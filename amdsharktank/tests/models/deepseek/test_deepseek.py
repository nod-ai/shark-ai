# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import unittest

import torch

from parameterized import parameterized

from amdsharktank.models.deepseek.toy_deepseek import generate
from amdsharktank.utils.llm_utils import (
    TorchInstance,
    llama_config_page_sizes,
    LlmRunner,
)
from amdsharktank.utils.evaluate import *
from amdsharktank.utils.testing import (
    is_mi300x,
    IreeVsEagerLLMTester,
    TempDirTestBase,
)


class DeepseekCrossEntropyTest(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float16, torch.float32),
            (torch.float32, torch.float32),
        ]
    )
    def testUnsharded(self, dtype_rest: torch.dtype, dtype_norm: torch.dtype):
        theta, cfg = generate(12345, dtype_rest=dtype_rest, dtype_norm=dtype_norm)
        model = TorchInstance(theta=theta, config=cfg)
        page_sizes = llama_config_page_sizes(model.config)

        ids = [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]]

        llm_batch = LlmRunner(
            instance=model,
            page_count=cfg.hp.block_count,
            page_sizes=page_sizes,
            block_stride=cfg.block_seq_stride,
            kv_cache_dtype="float16",
        )

        page_ids = [llm_batch.allocate(token_count=len(ids[0]))]
        logits, _ = llm_batch.prefill(ids, page_ids=page_ids)

        token_ids, _ = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=cfg.block_seq_stride,
        )
        ids = torch.as_tensor(token_ids[0][:-1])
        logits = torch.as_tensor(logits)[0, 1:]
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

        assert pytest.approx(9.7477, 1e-4) == cross_entropy


@pytest.mark.usefixtures("iree_flags", "device")
@is_mi300x
class DeepseekIreeVsEagerTest(TempDirTestBase):
    @pytest.mark.xfail(
        raises=AssertionError,
        reason="https://github.com/nod-ai/amdshark-ai/issues/1758",
        match="Outputs do not match for prefill batch index 0",
    )
    def testUnshardedToyIreeVsEager(self):
        theta, cfg = generate(12345)

        tester = IreeVsEagerLLMTester(
            work_dir=self._temp_dir,
            theta=theta,
            config=cfg,
            torch_device=self.device,
            iree_device=self.iree_device,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
        )
        tester.run_and_compare_iree_vs_eager()
