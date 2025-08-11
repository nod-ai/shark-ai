# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch


from sharktank.models.llm import *
from sharktank.models.llama.toy_llama import generate
from sharktank.utils.llm_utils import LlmInstance, TorchInstance, llama_config_page_size


class ToyLlamaTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)

        model = TorchInstance(theta=theta, config=config)
        page_size = llama_config_page_size(config)
        block_count = 128

        self._instance = LlmInstance(
            model_instance=model,
            page_size=page_size,
            block_seq_stride=config.block_seq_stride,
            block_count=block_count,
        )

    def testDecodeSequence(self):
        decoder = self._instance.make_decoder()

        # fmt: off
        expected = [ 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137, ]
        # fmt: on
        decoded = decoder.greedy_decode([[0]], steps=len(expected))
        assert all(torch.asarray(expected) == torch.asarray(decoded[0]))

    def testPrefillPerplexity(self):
        decoder = self._instance.make_perplexity_eval()

        # fmt: off
        seq = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137, ]
        # fmt: on
        all_match, score = decoder.prefill_cross_entropy([seq])[0]
        assert all_match == True
        torch.testing.assert_close(score, 0.583, atol=1e-2, rtol=1e-2)

    def testDecodePerplexity(self):
        decoder = self._instance.make_perplexity_eval()

        # fmt: off
        seq = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137, ]
        # fmt: on
        all_match, score = decoder.decode_cross_entropy([seq])[0]
        assert all_match == True
        torch.testing.assert_close(score, 0.583, atol=1e-2, rtol=1e-2)
