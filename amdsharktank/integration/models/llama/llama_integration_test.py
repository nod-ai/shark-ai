# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import unittest

import pytest
import torch

from pathlib import Path
from amdsharktank.layers import LlamaModelConfig
from amdsharktank.models.llm.testing import (
    clip_llm_block_count,
    make_random_token_sequences,
    run_perplexity_test_pipeline_parallel_eager_vs_eager,
)
from amdsharktank.types import (
    Dataset,
)
from amdsharktank.utils.tokenizer import load_tokenizer


@pytest.mark.expensive
def test_pruned_llama3_405b_f4_pipeline_parallel_eager_vs_eager_perplexity(
    deterministic_random_seed, test_data_dir: Path
):
    """Verify that a pipeline-parallel pruned (removed layers) variant of the 405B f4
    model produces the same perplexity as a the reference variant that is not
    pipeline-parallel.
    We don't care if the perplexity is high. Just that it is the same against the reference."""

    # We run specifically on CPU because as of 09/29/2025 there is no PyTorch
    # ROCm wheel release for gfx950 (MI350) AMD GPUs. It needs ROCm 7. Current is 6.4.
    # IREE is unable to compile on gfx942 (MI300X) the KV cache gather kernel.
    # More specifically iree_linalg_ext.gather.
    # error: 'hal.interface.binding.subspan' op F8E5M2 and F8E4M3FN types are not supported on gfx942 (MI-300) or older chipsets; try F8E5M2FNUZ or F8E4M3FNUZ instead.
    # This is probably expected if gfx942 really does not support this.
    # On CPU this test may take up to 5 minutes on a 64-core CPU.
    device = torch.device("cpu")

    batch_size = 4
    prune_to_block_count = 3
    pipeline_parallelism_size = 2

    parameters_path = (
        test_data_dir
        / "ossci-models/llama_3_1/405b/fp4/fp4_preshuffled_2025_09_12.irpa"
    )
    dataset = Dataset.load(parameters_path)

    reference_config = LlamaModelConfig.from_dataset(dataset)

    assert (
        reference_config.fake_quant == True
    ), "TODO: remove fake_quant fix below when it has the correct value. See https://github.com/nod-ai/amdshark-ai/issues/2388"
    reference_config.fake_quant = False

    assert reference_config.hp.rope_interleave_emb == False
    assert reference_config.fake_quant == False

    if reference_config.hp.vocab_size is None:
        # Get vocabulary size for the tokenizer as the IRPA does not have it.
        tokenizer_path = (
            test_data_dir / "ossci-models/llama_3_1/405b/fp4/tokenizer.json"
        )
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
