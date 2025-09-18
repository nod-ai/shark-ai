# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import unittest
import iree

from sharktank.models.llama.toy_llama import generate
from sharktank.utils.llm_artifacts import LlmArtifactBuilder, ExportConfig
from sharktank.utils.llm_utils import LlmInstance, TorchInstance, llama_config_page_size
from sharktank.utils.testing import is_cpu


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
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                raises=iree.compiler.CompilerToolError,
                strict=True,
                reason="https://github.com/iree-org/iree/issues/21889",
            ),
        ),
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

        page_size = llama_config_page_size(llama_config)
        block_count = 128

        self._instance = LlmInstance(
            model_instance=iree_instance,
            page_size=page_size,
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
