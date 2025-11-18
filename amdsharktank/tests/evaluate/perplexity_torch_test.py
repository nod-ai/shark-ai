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

from amdsharktank.evaluate import perplexity_torch
from amdsharktank.utils.testing import (
    is_llama_8b,
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


if __name__ == "__main__":
    unittest.main()
