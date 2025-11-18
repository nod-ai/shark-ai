# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import unittest

import math
import pytest
import torch


from pathlib import Path
from amdsharktank.models.llm.llm import PagedLlmModelV1
from amdsharktank.models.llama.toy_llama import generate
from amdsharktank.utils import chdir
from amdsharktank.utils.export_artifacts import IreeCompileException
from amdsharktank.utils.testing import (
    is_mi300x,
    IreeVsEagerLLMTester,
    TempDirTestBase,
)


def generate_args(
    requests: list[list[int]],
    page_ids: list[list[int]],
    block_seq_stride: int,
    offsets: list[int] | None = None,
):
    bs = len(requests)
    if offsets is None:
        offsets = [0] * bs

    max_len = max(len(t) for t in requests)
    seq_lens = [len(t) + s for t, s in zip(requests, offsets)]
    max_ctx = max(seq_lens)

    max_len = math.ceil((max_len - 1) / block_seq_stride) * block_seq_stride
    max_blocks = math.ceil((max_ctx - 1) / block_seq_stride)

    tokens = torch.zeros((bs, max_len), dtype=torch.int64)
    for i, ids in enumerate(requests):
        tokens[i, : len(ids)] = torch.asarray(ids)

    offsets = torch.asarray(offsets, dtype=torch.int64)
    seq_lens = torch.asarray(seq_lens, dtype=torch.int64)
    seq_block_ids = torch.zeros((bs, max_blocks), dtype=torch.int64)
    for i, ids in enumerate(page_ids):
        seq_block_ids[i, : len(ids)] = torch.asarray(ids[:max_blocks])

    return {
        "tokens": tokens,
        "start_positions": offsets,
        "seq_lens": seq_lens,
        "seq_block_ids": seq_block_ids,
    }


class CrossEntropyTest(unittest.TestCase):
    def testShortPrefill(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        # fmt: off
        tokens = [[0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]]
        # fmt: on
        blocks = [[1]]
        expected_ce = [0.583]

        kwargs = generate_args(
            tokens, page_ids=blocks, block_seq_stride=config.block_seq_stride
        )

        cache_state = model.cache.allocate(page_count=64)

        logits = model.prefill(
            cache_state=cache_state,
            **kwargs,
        )

        tokens = tokens[0]
        logits = logits[0]
        seq_len = len(tokens)
        expected = tokens[1:seq_len]
        expected = torch.asarray(expected, dtype=torch.int64)
        logits = logits[: seq_len - 1, :].to(torch.float32)

        cross_entropy = torch.nn.functional.cross_entropy(logits, expected)
        assert pytest.approx(expected_ce[0], 1e-2) == cross_entropy

    def testPrefill(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        # fmt: off
        tokens = [[0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137,
                   163, 117, 72, 250, 118, 127, 214, 184, 194, 23, 151, 186, 160, 35, 59, 58]]
        # fmt: on
        blocks = [[1, 2]]
        expected_ce = [0.629]

        kwargs = generate_args(
            tokens, page_ids=blocks, block_seq_stride=config.block_seq_stride
        )

        cache_state = model.cache.allocate(page_count=64)

        logits = model.prefill(
            cache_state=cache_state,
            **kwargs,
        )

        tokens = tokens[0]
        logits = logits[0]
        seq_len = len(tokens)
        expected = tokens[1:seq_len]
        expected = torch.asarray(expected, dtype=torch.int64)
        logits = logits[: seq_len - 1, :].to(torch.float32)

        cross_entropy = torch.nn.functional.cross_entropy(logits, expected)
        assert pytest.approx(expected_ce[0], 1e-2) == cross_entropy

    def testOffsetPrefill(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        # fmt: off
        tokens0 = [[0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]]
        tokens1 = [[ 163, 117, 72, 250, 118, 127, 214, 184, 194, 23, 151, 186, 160, 35, 59, 58]]
        # fmt: on
        blocks0 = [[1]]
        blocks1 = [[1, 2]]
        expected_ce = [0.629]

        cache_state = model.cache.allocate(page_count=64)

        kwargs = generate_args(
            tokens0,
            page_ids=blocks0,
            block_seq_stride=config.block_seq_stride,
            offsets=[0],
        )
        print(kwargs)
        logits0 = model.prefill(
            cache_state=cache_state,
            **kwargs,
        )

        kwargs = generate_args(
            tokens1,
            page_ids=blocks1,
            block_seq_stride=config.block_seq_stride,
            offsets=[config.block_seq_stride],
        )
        print(kwargs)
        logits1 = model.prefill(
            cache_state=cache_state,
            **kwargs,
        )

        tokens = tokens0[0] + tokens1[0]
        logits = torch.concatenate((logits0, logits1), dim=1)[0]

        seq_len = len(tokens)
        expected = tokens[1:seq_len]
        expected = torch.asarray(expected, dtype=torch.int64)
        logits = logits[: seq_len - 1, :].to(torch.float32)

        cross_entropy = torch.nn.functional.cross_entropy(logits, expected)
        assert pytest.approx(expected_ce[0], 1e-2) == cross_entropy


@pytest.mark.usefixtures("iree_flags", "device")
@is_mi300x
class LlamaIreeVsEagerTest(TempDirTestBase):
    @pytest.mark.xfail(
        raises=IreeCompileException,
        reason="https://github.com/iree-org/iree/issues/21462, https://github.com/nod-ai/amdshark-ai/issues/1758",
    )
    def testUnshardedToyIreeVsEager(self):
        theta, config = generate(12345)

        tester = IreeVsEagerLLMTester(
            work_dir=self._temp_dir,
            theta=theta,
            config=config,
            torch_device=self.device,
            iree_device=self.iree_device,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
        )
        tester.run_and_compare_iree_vs_eager()


@pytest.mark.expensive
def test_import_llama3_8B_instruct(tmp_path: Path):
    from amdsharktank.tools.import_hf_dataset_from_hub import main

    irpa_path = tmp_path / "model.irpa"
    main(
        [
            "--revision=0e9e39f249a16976918f6564b8830bc894c89659",
            f"--output-irpa-file={irpa_path}",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    )
    assert irpa_path.exists()


@pytest.mark.expensive
def test_import_llama3_8B_instruct_from_preset(tmp_path: Path):
    from amdsharktank.tools.import_hf_dataset_from_hub import main

    irpa_path = tmp_path / "llama3.1/8b/instruct/f16/model.irpa"
    tokenizer_path = tmp_path / "llama3.1/8b/instruct/f16/tokenizer.json"
    tokenizer_config_path = tmp_path / "llama3.1/8b/instruct/f16/tokenizer_config.json"
    with chdir(tmp_path):
        main(
            [
                "--preset=meta_llama3_1_8b_instruct_f16",
            ]
        )
    assert irpa_path.exists()
    assert tokenizer_path.exists()
    assert tokenizer_config_path.exists()
