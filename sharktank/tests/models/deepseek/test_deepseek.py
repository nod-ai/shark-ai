# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import torch

from sharktank.models.llm import *
from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.utils.load_llm import *
from sharktank.utils.evaluate import *
from sharktank.utils.testing import is_mi300x, IreeVsEagerLLMTest, xfail


@pytest.mark.usefixtures("get_iree_flags")
# @is_mi300x  # TODO: This is not working for some reason
class DeepseekTest(IreeVsEagerLLMTest):
    def testCrossEntropy(self):
        theta, config = generate(12345)
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

    @xfail(
        raises=AssertionError,
        reason="https://github.com/iree-org/iree/issues/21087",
        strict=True,
        match="Outputs do not match for batch index 0:",
    )
    def testUnshardedToySizedModelIREEVsEager(self):
        theta, config = generate(12345)

        raw_token_ids = [
            [1, 2, 3, 4],
            [9, 8, 7, 6],
            [3, 5, 2, 1],
        ]

        self.setup_variables(
            theta=theta,
            config=config,
            raw_token_ids=raw_token_ids,
            model_name="toy_deepseek",
            skip_decode=True,
        )
        self.run_iree_vs_eager()
