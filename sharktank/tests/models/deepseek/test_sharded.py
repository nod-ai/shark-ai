# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest
from copy import deepcopy

import torch

import unittest
from copy import deepcopy

import torch

from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.models.llm import PagedLlmModelV1
from sharktank.types.sharding import shard_theta
from sharktank.utils.evaluate import pad_tokens
from sharktank.utils.load_llm import TorchGenerator
from sharktank.utils.create_cache import *


@pytest.mark.skip(
    reason="Deepseek support will be added soon",
)
class DeepseekShardedTest(unittest.TestCase):
    def testTensorParallelToySizedModelEagerVsUnsharded(self):
        theta, config = generate(12345)
        tensor_parallelism_size = 2

        sharded_config = deepcopy(config)
        sharded_config.tensor_parallelism_size = tensor_parallelism_size
        sharded_theta = shard_theta(theta=theta, config=sharded_config)

        reference_model = PagedLlmModelV1(theta=theta, config=config)
        target_model = PagedLlmModelV1(theta=sharded_theta, config=sharded_config)

        ids = [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]]
        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)

        reference_generator = TorchGenerator(reference_model)
        reference_batch = reference_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        reference_batch.prefill()
        reference_logits = reference_batch.prefill_logits

        target_generator = TorchGenerator(target_model)
        target_batch = target_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        target_batch.prefill()
        target_logits = target_batch.prefill_logits

        torch.testing.assert_close(
            target_logits, reference_logits, atol=2e-4, rtol=2e-2
        )
