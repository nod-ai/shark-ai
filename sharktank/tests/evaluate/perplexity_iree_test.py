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
from sharktank.utils.export_artifacts import (
    IreeCompileException,
)

from sharktank.evaluate import perplexity_iree
from sharktank.utils.testing import (
    is_mi300x,
    is_nightly,
    is_deepseek,
    is_llama_8b,
    is_sharded,
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
        # NOTE: --use-attention-mask is required until https://github.com/nod-ai/shark-ai/issues/1202 is solved
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

        if self.tensor_parallelism_size * self.pipeline_parallelism_size > 1:
            self.argv.append(f"--use-attention-mask")
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

    @is_nightly
    def test_llama3_8B_f16_tp2(self):
        # Llama 3.1 8B fp16 tensor parallelism
        self.model_name = "llama3_8B_f16_iree"
        self.irpa_file = self.llama3_8b_f16_tp2_model
        self.tokenizer = self.llama3_8b_tokenizer
        self.tensor_parallelism_size = 2

        self.prepare_argv()
        self.run_and_check_perplexity()

    @is_nightly
    def test_llama3_8B_f16_pp2(self):
        # Llama 3.1 8B fp16 pipepiline parallelism
        self.model_name = "llama3_8B_f16_iree"
        self.irpa_file = self.llama3_8b_f16_model
        self.tokenizer = self.llama3_8b_tokenizer
        self.pipeline_parallelism_size = 2

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
                f"--attention-dtype=float8_e4m3fnuz",
                f"--activation-dtype=bfloat16",
                f"--kv-cache-dtype=float8_e4m3fnuz",
                "--use-hf",
                "--use-attention-mask",
                "--attention-kernel=sharktank",
            )
        )
        self.run_and_check_perplexity()

    @is_sharded
    def test_llama3_70B_f16_pp8(self):
        # Llama 3.1 70B fp16 non-decomposed
        self.model_name = "llama3_70B_f16_iree"
        self.irpa_file = self.llama3_70b_f16_model
        self.tokenizer = self.llama3_70b_tokenizer
        self.pipeline_parallelism_size = 8

        self.prepare_argv()
        self.run_and_check_perplexity()

    @pytest.mark.skip(reason="70B fp8 model unavailable")
    @is_sharded
    def test_llama3_70B_f8_pp8(self):
        # Llama 3.1 70B fp8 non-decomposed
        self.model_name = "llama3_70B_f8_iree"
        self.irpa_file = self.llama3_70b_f8_model
        self.tokenizer = self.llama3_70b_tokenizer
        self.pipeline_parallelism_size = 8

        self.prepare_argv()
        self.run_and_check_perplexity()

    @pytest.mark.xfail(
        raises=IreeCompileException,
        reason="https://github.com/iree-org/iree/issues/21068",
        strict=True,
        match="failed to solve for affinity analysis",
    )
    @is_sharded
    def test_llama3_405B_f16_tp8(self):
        # Llama 3.1 405B fp16 non-decomposed
        self.model_name = "llama3_405B_f16_iree"
        self.irpa_file = self.llama3_405b_f16_tp8_model
        self.tokenizer = self.llama3_405b_tokenizer
        self.tensor_parallelism_size = 8

        self.prepare_argv()
        self.run_and_check_perplexity()

    @pytest.mark.skip(reason="405B fp8 model unavailable")
    @is_sharded
    def test_llama3_405B_f8_tp8(self):
        # Llama 3.1 405B fp8 non-decomposed
        self.model_name = "llama3_405B_f8_iree"
        self.irpa_file = self.llama3_405b_f8_tp8_model
        self.tokenizer = self.llama3_405b_tokenizer
        self.tensor_parallelism_size = 8

        self.prepare_argv()
        self.run_and_check_perplexity()

    @is_deepseek
    def test_deepseek_v3(self):
        # DeepSeek v3
        self.model_name = "deepseek_v3_iree"
        self.irpa_file = self.deepseek_v3_model
        self.tokenizer = self.deepseek_v3_tokenizer
        self.delta = 10

        self.prepare_argv(extra_args=(f"--use-toy-model",))
        self.run_and_check_perplexity()

    @pytest.mark.xfail(
        raises=IreeCompileException,
        reason="https://github.com/iree-org/iree/issues/20914",
        strict=True,
        match="operation destroyed but still has uses",
    )
    @is_nightly
    def test_deepseek_v3_tp2(self):
        # DeepSeek v3 tensor parallelism
        self.model_name = "deepseek_v3_iree"
        self.irpa_file = self.deepseek_v3_tp2_model
        self.tokenizer = self.deepseek_v3_tokenizer
        self.tensor_parallelism_size = 2
        self.delta = 10

        self.prepare_argv(extra_args=("--use-toy-model",))
        self.run_and_check_perplexity()

    @is_deepseek
    def test_deepseek_v3_pp2(self):
        # DeepSeek v3 pipeline parallelism
        self.model_name = "deepseek_v3_iree"
        self.irpa_file = self.deepseek_v3_model
        self.tokenizer = self.deepseek_v3_tokenizer
        self.pipeline_parallelism_size = 2
        # TODO: https://github.com/nod-ai/shark-ai/issues/1855
        # Showed up after https://github.com/nod-ai/shark-ai/pull/1735
        self.delta = 10

        self.prepare_argv(extra_args=(f"--use-toy-model",))
        self.run_and_check_perplexity()


if __name__ == "__main__":
    unittest.main()
