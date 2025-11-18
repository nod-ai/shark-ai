# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import unittest
import iree

from amdsharktank.models.llama.testing import quantize_theta_to_fp4
from amdsharktank.models.llama.toy_llama import generate, generate2
from amdsharktank.types import (
    DynamicFp4BlockQuantizer,
    InferenceTensorTransforms,
)
from amdsharktank.models.llm.testing import (
    make_random_token_sequences,
    run_perplexity_test_pipeline_parallel_eager_vs_eager,
)
from amdsharktank.utils.llm_artifacts import LlmArtifactBuilder, ExportConfig
from amdsharktank.utils.llm_utils import (
    LlmInstance,
    TorchInstance,
    llama_config_page_sizes,
)
from amdsharktank.utils.testing import is_cpu


def get_iree_compile_flags(self):
    flags = []

    if self.iree_hal_target_device is not None:
        flags.append(f"--iree-hal-target-device={self.iree_hal_target_device}")

    if self.iree_hal_target_device == "local":
        flags.append("--iree-hal-local-target-device-backends=llvm-cpu")

    if self.iree_hip_target is not None:
        flags.append(f"--iree-hip-target={self.iree_hip_target}")

    return flags


class ToyLlamaTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)

        model = TorchInstance(theta=theta, config=config)
        page_sizes = llama_config_page_sizes(config)
        block_count = 128

        self._instance = LlmInstance(
            model_instance=model,
            page_sizes=page_sizes,
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
        result = decoder.prefill_cross_entropy([seq])[0]
        assert result.valid
        torch.testing.assert_close(result.score, 0.583, atol=1e-2, rtol=1e-2)

    def testDecodePerplexity(self):
        decoder = self._instance.make_perplexity_eval()

        # fmt: off
        seq = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137, ]
        # fmt: on
        result = decoder.decode_cross_entropy([seq])[0]
        assert result.valid
        torch.testing.assert_close(result.score, 0.583, atol=1e-2, rtol=1e-2)


@pytest.mark.usefixtures("iree_flags")
@is_cpu
@pytest.mark.parametrize(
    "use_extend_attention",
    [
        True,
        False,
    ],
)
class TestToyLlamaIree:
    @pytest.fixture(scope="function", autouse=True)
    def setUp(self, use_extend_attention):
        torch.set_default_dtype(torch.float32)
        theta, llama_config = generate(12345)
        llm_artifact = LlmArtifactBuilder(theta=theta, llama_config=llama_config)

        export_config = ExportConfig(
            logits_normalization="log_softmax",
            use_extend_attention=use_extend_attention,
        )
        llm_artifact.export(export_config)

        compiler_flags = get_iree_compile_flags(self)
        llm_artifact.compile(compiler_flags)

        iree_instance = llm_artifact.instance([self.iree_device])

        page_sizes = llama_config_page_sizes(llama_config)
        block_count = 128

        self._instance = LlmInstance(
            model_instance=iree_instance,
            page_sizes=page_sizes,
            block_seq_stride=llama_config.block_seq_stride,
            block_count=block_count,
        )

    def testPrefillPerplexity(self):
        decoder = self._instance.make_perplexity_eval()

        # fmt: off
        seq = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137, ]
        # fmt: on
        result = decoder.prefill_cross_entropy([seq])[0]
        assert result.valid
        torch.testing.assert_close(result.score, 0.583, atol=1e-2, rtol=1e-2)

    def testDecodePerplexity(self):
        decoder = self._instance.make_perplexity_eval()

        # fmt: off
        seq = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137, ]
        # fmt: on
        result = decoder.decode_cross_entropy([seq])[0]
        assert result.valid
        torch.testing.assert_close(result.score, 0.583, atol=1e-2, rtol=1e-2)


def test_toy_llama3_f4_pipeline_parallel_eager_vs_eager_perplexity(
    deterministic_random_seed,
):
    """Verify that a pipeline-parallel toy Llama 3 model produces the
    same perplexity as a the reference variant that is not pipeline-parallel."""

    # We run specifically on CPU because as of 09/29/2025 there is no PyTorch
    # ROCm wheel release for gfx950 (MI350) AMD GPUs. It needs ROCm 7. Current is 6.4.
    # IREE is unable to compile on gfx942 (MI300X) the KV cache gather kernel.
    # More specifically iree_linalg_ext.gather.
    # error: 'hal.interface.binding.subspan' op F8E5M2 and F8E4M3FN types are not supported on gfx942 (MI-300) or older chipsets; try F8E5M2FNUZ or F8E4M3FNUZ instead.
    # This is probably expected if gfx942 really does not support this.
    device = torch.device("cpu")

    batch_size = 2
    pipeline_parallelism_size = 2

    reference_theta, reference_config = generate2(seed=0)

    reference_config.kv_cache_dtype = torch.float8_e4m3fn
    reference_config.device = device
    reference_theta = reference_theta.transform(
        InferenceTensorTransforms.to_device(device)
    )
    reference_theta = quantize_theta_to_fp4(
        reference_theta,
        quantizer=DynamicFp4BlockQuantizer(
            block_size=batch_size, use_amdsharktank_kernel=False
        ),
    )

    tokens = make_random_token_sequences(
        num_sequences=batch_size,
        min_tokens_per_sequence=3,
        max_tokens_per_sequence=3,
        vocabulary_size=reference_config.hp.vocab_size,
    )
    run_perplexity_test_pipeline_parallel_eager_vs_eager(
        reference_theta=reference_theta,
        reference_config=reference_config,
        tokens=tokens,
        pipeline_parallelism_size=pipeline_parallelism_size,
    )
