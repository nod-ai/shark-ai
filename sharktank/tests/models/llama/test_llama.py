# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import unittest

import pytest
import torch

from copy import deepcopy
from pathlib import Path
from sharktank.layers import LlamaModelConfig
from sharktank.models.llm.llm import PagedLlmModelV1
from sharktank.models.llm.testing import (
    clip_llm_block_count,
    make_random_token_sequences,
    run_perplexity_test_pipeline_parallel_eager_vs_eager,
)
from sharktank.models.llama.toy_llama import generate
from sharktank.types import (
    Dataset,
)
from sharktank.utils.export_artifacts import IreeCompileException
from sharktank.utils.testing import (
    is_mi300x,
    IreeVsEagerLLMTester,
    TempDirTestBase,
)
from sharktank.utils.tokenizer import load_tokenizer


class CrossEntropyTest(unittest.TestCase):
    def testUnsharded(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
        seq_len = len(ids)

        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        ids = ids + [0] * padding

        ids = torch.asarray([ids], dtype=torch.int64)
        block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        logits = model.prefill(
            tokens=ids,
            seq_lens=torch.tensor([seq_len]),
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding
        ids = ids[:, :seq_len]
        logits = logits[:, :seq_len, :]

        ids = ids[0, 1:]
        logits = logits[0, :-1].to(torch.float32)
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)
        assert pytest.approx(0.583, 1e-2) == cross_entropy


@pytest.mark.expensive
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test is too slow on CPU. Could take up to 5 minutes on a 64 core CPU.",
)
def test_pruned_llama3_405b_f4_pipeline_parallel_eager_vs_eager_perplexity(
    deterministic_random_seed, model_artifacts: dict[str, str]
):
    """Verify that a pipeline-parallel pruned (removed layers) variant of the 405B f4
    model produces the same perplexity as a the reference variant that is not
    pipeline-parallel.
    We don't care if the perplexity is high. Just that it is the same against the reference."""
    device = torch.device("cuda")
    batch_size = 4
    prune_to_block_count = 3
    pipeline_parallelism_size = 2

    parameters_path = model_artifacts["llama3_1_405b_instruct_f4_model_path"]
    dataset = Dataset.load(parameters_path)

    reference_config = LlamaModelConfig.from_dataset(dataset)
    # TODO: remove when the IRPA has the correct value
    reference_config.hp.rope_interleave_emb = False
    if reference_config.hp.vocab_size is None:
        # Get vocabulary size for the tokenizer as the IRPA does not have it.
        tokenizer_path = model_artifacts["llama3_1_405b_tokenizer_path"]
        tokenizer = load_tokenizer(Path(tokenizer_path).parent)
        reference_config.hp.vocab_size = tokenizer.vocab_size
    reference_config.kv_cache_dtype = torch.float8_e4m3fn

    reference_config.device = device
    reference_config.hp.block_count = prune_to_block_count
    reference_theta = dataset.root_theta
    reference_theta, reference_config = clip_llm_block_count(
        reference_theta, reference_config, block_count=prune_to_block_count
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


@pytest.mark.usefixtures("iree_flags", "device")
@is_mi300x
class LlamaIreeVsEagerTest(TempDirTestBase):
    @pytest.mark.xfail(
        raises=IreeCompileException,
        reason="https://github.com/iree-org/iree/issues/21462, https://github.com/nod-ai/shark-ai/issues/1758",
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
    from sharktank.tools.import_hf_dataset_from_hub import main

    irpa_path = tmp_path / "model.irpa"
    main(
        [
            "--revision=0e9e39f249a16976918f6564b8830bc894c89659",
            f"--output-irpa-file={irpa_path}",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    )
    assert irpa_path.exists()
