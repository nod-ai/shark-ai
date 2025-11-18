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

from amdsharktank.evaluate import perplexity_iree
from amdsharktank.utils.testing import (
    is_mi300x,
    is_llama_8b,
)


@pytest.mark.usefixtures(
    "model_artifacts",
    "iree_flags",
    "tensor_parallelism_size",
    "baseline_perplexity_scores",
    "batch_size",
)
@is_mi300x
class PerplexityTest(unittest.TestCase):
    def setUp(self):
        self.current_perplexity_all = {}
        self.delta = 5e-1
        self.tensor_parallelism_size = 1
        self.pipeline_parallelism_size = 1
        with open(self.baseline_perplexity_scores, "r") as f:
            self.baseline_perplexity = json.load(f)
        self.iree_devices = (
            [self.iree_device]
            if isinstance(self.iree_device, str)
            else self.iree_device
        )

    def prepare_argv(self, extra_args: Iterable | None = None):
        self.argv = [
            f"--irpa-file={self.irpa_file}",
            f"--tokenizer-config-json={self.tokenizer}",
            f"--iree-hal-target-device={self.iree_hal_target_device}",
            f"--iree-hip-target={self.iree_hip_target}",
            f"--tensor-parallelism-size={self.tensor_parallelism_size}",
            f"--pipeline-parallelism-size={self.pipeline_parallelism_size}",
            f"--num-prompts={self.batch_size}",
        ]
        self.argv.extend(f"--iree-device={device}" for device in self.iree_devices)

        if extra_args:
            self.argv.extend(extra_args)

    def run_and_check_perplexity(self):
        current_perplexity = perplexity_iree.main(self.argv)
        baseline_perplexity = self.baseline_perplexity[self.model_name]

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

    @is_llama_8b
    def test_llama3_8B_f16(self):
        # Llama 3.1 8B fp16 non-decomposed
        self.model_name = "llama3_8B_f16_iree"
        self.irpa_file = self.llama3_8b_f16_model
        self.tokenizer = self.llama3_8b_tokenizer

        self.prepare_argv()
        self.run_and_check_perplexity()

    @is_llama_8b
    def test_llama3_8B_f8(self):
        # Llama 3.1 8B fp8 non-decomposed
        self.model_name = "llama3_8B_f8_iree"
        self.irpa_file = self.llama3_8b_f8_model
        self.tokenizer = self.llama3_8b_tokenizer

        self.prepare_argv(
            extra_args=(
                "--use-hf",
                "--attention-kernel=amdsharktank",
            )
        )
        self.run_and_check_perplexity()


if __name__ == "__main__":
    unittest.main()
