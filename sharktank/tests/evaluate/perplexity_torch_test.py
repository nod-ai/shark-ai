# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Iterable
import unittest
import pytest
import json
import numpy as np
import gc

from sharktank.evaluate import perplexity_torch
from sharktank.utils.testing import (
    is_nightly,
    is_llama_8b,
    is_deepseek,
    is_sharded,
)


@pytest.mark.usefixtures(
    "model_artifacts",
    "tensor_parallelism_size",
    "baseline_perplexity_scores",
    "batch_size",
    "device",
)
class PerplexityTest(unittest.TestCase):
    def setUp(self):
        self.current_perplexity_all = {}
        self.delta = 5e-1
        self.tensor_parallelism_size = 1
        self.pipeline_parallelism_size = 1
        with open(self.baseline_perplexity_scores, "r") as f:
            self.baseline_perplexity = json.load(f)

    def prepare_argv(self, extra_args: Iterable | None = None):
        self.argv = [
            f"--irpa-file={self.irpa_file}",
            f"--tokenizer-config-json={self.tokenizer}",
            f"--num-prompts={self.batch_size}",
            f"--device={self.device}",
            f"--tensor-parallelism-size={self.tensor_parallelism_size}",
            f"--pipeline-parallelism-size={self.pipeline_parallelism_size}",
            "--use-attention-mask",
        ]
        if extra_args:
            self.argv.extend(extra_args)

    def run_and_check_perplexity(self):
        baseline_perplexity = self.baseline_perplexity[self.model_name]
        current_perplexity = perplexity_torch.main(self.argv)

        baseline_mean_perplexity = round(
            np.mean(baseline_perplexity["perplexities"][0 : self.batch_size]), 6
        )
        current_mean_perplexity = round(current_perplexity["mean_perplexity"], 6)

        perplexity_difference = current_mean_perplexity - baseline_mean_perplexity

        self.assertAlmostEqual(
            baseline_mean_perplexity,
            current_mean_perplexity,
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )
        gc.collect()

    @is_llama_8b
    def test_llama3_8B_f16(self):
        # Llama 3.1 8B non-decomposed
        self.model_name = "llama3_8B_f16_torch"
        self.irpa_file = self.llama3_8b_f16_model
        self.tokenizer = self.llama3_8b_tokenizer

        self.prepare_argv()
        self.run_and_check_perplexity()

    @is_deepseek
    def test_deepseek_v3(self):
        # DeepSeek v3 unsharded toy test
        self.model_name = "deepseek_v3_torch"
        self.irpa_file = self.deepseek_v3_model
        self.tokenizer = self.deepseek_v3_tokenizer

        self.prepare_argv(extra_args=("--use-toy-model",))
        self.run_and_check_perplexity()

    @is_nightly
    def test_deepseek_v3_tp(self):
        # DeepSeek v3 tensor parallelism
        self.model_name = "deepseek_v3_torch"
        self.irpa_file = self.deepseek_v3_tp2_model
        self.tokenizer = self.deepseek_v3_tokenizer
        self.tensor_parallelism_size = 2
        self.delta = 12

        self.prepare_argv(extra_args=("--use-toy-model",))
        self.run_and_check_perplexity()

    @is_nightly
    def test_deepseek_v3_pp(self):
        # DeepSeek v3 pipeline parallelism
        self.model_name = "deepseek_v3_torch"
        self.irpa_file = self.deepseek_v3_model
        self.tokenizer = self.deepseek_v3_tokenizer
        self.pipeline_parallelism_size = 2

        self.prepare_argv(extra_args=("--use-toy-model",))
        self.run_and_check_perplexity()


if __name__ == "__main__":
    unittest.main()
