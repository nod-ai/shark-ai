# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest
import json
import numpy as np

from sharktank.evaluate import perplexity_iree
from sharktank.utils.testing import is_mi300x


skipif_run_quick_llama_test = pytest.mark.skipif(
    'not config.getoption("run-nightly-llama-tests")',
    reason="Run large tests if --run-nightly-llama-tests is passed",
)


@pytest.mark.usefixtures(
    "get_model_artifacts",
    "get_iree_flags",
    "tensor_parallelism_size",
    "baseline_perplexity_scores",
    "batch_size",
)
@is_mi300x
class PerplexityTest(unittest.TestCase):
    def setUp(self):
        self.current_perplexity_all = {}
        self.delta = 5e-1
        self.tensor_parallelism_size = 8
        with open(self.baseline_perplexity_scores, "r") as f:
            self.baseline_perplexity = json.load(f)

    def test_llama3_8B_f16_decomposed(self):

        # Llama 3.1 8B decomposed

        model_name = "llama3_8B_f16_decomposed_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_8b_f16_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
                f"--tensor-parallelism-size=1",
                f"--attention-kernel=decomposed",
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
    @pytest.mark.xfail(reason="Compile Error")
    def test_llama3_8B_f16(self):

        # Llama 3.1 8B non-decomposed

        model_name = "llama3_8B_f16_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_8b_f16_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
                f"--tensor-parallelism-size=1",
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
    @pytest.mark.xfail(reason="Compile Error")
    def test_llama3_8B_fp8_decomposed(self):

        # Llama 3.1 8B decomposed

        model_name = "llama3_8B_fp8_decomposed_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_8b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
                f"--tensor-parallelism-size=1",
                f"--attention-kernel=decomposed",
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
    @pytest.mark.xfail(reason="Compile Error")
    def test_llama3_8B_fp8(self):

        # Llama 3.1 8B non-decomposed

        model_name = "llama3_8B_fp8_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_8b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_8b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
                f"--tensor-parallelism-size=1",
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
    @pytest.mark.xfail(
        reason="Sharding is unsupported",
    )
    def test_llama3_405B_f16_decomposed(self):

        # Llama 3.1 405B decomposed

        model_name = "llama3_405B_f16_decomposed_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_405b_f16_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
                f"--attention-kernel=decomposed",
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
    @pytest.mark.xfail(reason="Compile Error")
    def test_llama3_405B_f16(self):

        # Llama 3.1 405B non-decomposed

        model_name = "llama3_405B_f16_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_405b_f16_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
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

    @skipif_run_quick_llama_test
    @pytest.mark.xfail(reason="Compile Error")
    def test_llama3_405B_fp8_decomposed(self):

        # Llama 3.1 405B decomposed

        model_name = "llama3_405B_fp8_decomposed_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_405b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
                f"--tensor-parallelism-size={self.tensor_parallelism_size}",
                f"--attention-kernel=decomposed",
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
    @pytest.mark.xfail(reason="Compile Error")
    def test_llama3_405B_fp8(self):

        # Llama 3.1 405B non-decomposed

        model_name = "llama3_405B_fp8_iree"
        baseline_perplexity = self.baseline_perplexity[model_name]

        current_perplexity = perplexity_iree.main(
            [
                f"--irpa-file={self.llama3_405b_fp8_model}",
                f"--tokenizer-config-json={self.llama3_405b_tokenizer}",
                f"--iree-device={self.iree_device}",
                f"--iree-hal-target-device={self.iree_hal_target_device}",
                f"--iree-hip-target={self.iree_hip_target}",
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
