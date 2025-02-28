# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest
import json
import numpy as np

from sharktank.evaluate import perplexity_torch

skipif_run_quick_llama_test = pytest.mark.skipif(
    'not config.getoption("run-nightly-llama-tests")',
    reason="Run large tests if --run-nightly-llama-tests is passed",
)


@pytest.mark.usefixtures(
    "get_model_artifacts",
    "tensor_parallelism_size",
    "baseline_perplexity_scores",
    "batch_size",
)
class PerplexityTest(unittest.TestCase):
    def setUp(self):
        self.current_perplexity_all = {}
        self.delta = 5e-1
        self.tensor_parallelism_size = 8
        with open(self.baseline_perplexity_scores, "r") as f:
            self.baseline_perplexity = json.load(f)

    @skipif_run_quick_llama_test
    def test_llama3_8B_f16(self):

        # Llama 3.1 8B non-decomposed

        model_name = "llama3_8B_f16_torch"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_8b_f16_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--attention-kernel=torch",
                f"--num-prompts={self.batch_size}",
            ]
        )

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

    @skipif_run_quick_llama_test
    def test_llama3_8B_f8(self):

        # Llama 3.1 8B non-decomposed

        model_name = "llama3_8B_f8_torch"
        baseline_perplexity = self.baseline_perplexity[model_name]

        batch_size = 8

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_8b_f8_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--attention-kernel=torch",
                f"--num-prompts={batch_size}",
                f"--attention-dtype=bfloat16",
                f"--activation-dtype=bfloat16",
                "--use-hf",
                "--fake-quant",
            ]
        )

        baseline_mean_perplexity = round(
            np.mean(baseline_perplexity["perplexities"][0:batch_size]), 6
        )
        current_mean_perplexity = round(current_perplexity["mean_perplexity"], 6)

        perplexity_difference = current_mean_perplexity - baseline_mean_perplexity

        self.assertAlmostEqual(
            baseline_mean_perplexity,
            current_mean_perplexity,
            delta=self.delta,
            msg=f"Current perplexity deviates baseline by {perplexity_difference}",
        )

    @pytest.mark.xfail(
        reason="Non-decomposed attention is not supported yet",
    )
    @skipif_run_quick_llama_test
    def test_llama3_405B_f16(self):

        # Llama 3.1 405B non-decomposed

        model_name = "llama3_405B_f16_torch"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_405b_f16_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
                f"--attention-kernel=torch",
                f"--num-prompts={self.batch_size}",
            ]
        )

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

    @pytest.mark.xfail(
        reason="Non-decomposed attention is not supported yet",
    )
    @skipif_run_quick_llama_test
    def test_llama3_405B_f8(self):

        # Llama 3.1 405B non-decomposed

        model_name = "llama3_405B_f8_torch"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_torch.main(
            [
                f"--irpa-file={self.llama3_405b_f8_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
                f"--attention-kernel=torch",
                f"--num-prompts={self.batch_size}",
            ]
        )

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


if __name__ == "__main__":
    unittest.main()
