# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import torch

from copy import deepcopy
from amdsharktank.types import (
    dtype_to_serialized_name,
    InferenceTensorTransforms,
    Theta,
)
from amdsharktank.types.pipelining import pipeline_parallelize_llm_theta
from amdsharktank.layers import LlamaModelConfig, ParallelismConfig
from amdsharktank.utils.llm_utils import (
    IreeInstance,
    LlmPerplexityEval,
    LlmInstance,
    TorchInstance,
    llama_config_page_sizes,
    minimum_required_kv_cache_page_count_for_batch,
)
from amdsharktank.utils.testing import assert_tensor_close
from typing import Callable


def make_random_token_sequences(
    num_sequences: int,
    min_tokens_per_sequence: int,
    max_tokens_per_sequence: int,
    vocabulary_size: int,
) -> list[list[int]]:
    assert min_tokens_per_sequence > 1
    seq_lens = torch.randint(
        size=[num_sequences],
        low=min_tokens_per_sequence,
        high=max_tokens_per_sequence + 1,
        dtype=torch.int64,
    )
    max_seq_len = int(seq_lens.max())
    token_ids = torch.randint(
        low=0,
        high=vocabulary_size,
        size=[num_sequences, max_seq_len],
        dtype=torch.int64,
    )
    token_ids_list = token_ids.tolist()
    token_ids_list = [
        seq_token_ids[: int(seq_len)]
        for seq_len, seq_token_ids in zip(seq_lens, token_ids_list)
    ]
    return token_ids_list


class LlmPerplexityCompare:
    """Compare the perplexity of one implementation against another.
    Checks that the 2 models' results do not deviate from each other."""

    def __init__(
        self,
        make_target_model: Callable[[], LlmInstance],
        make_reference_model: Callable[[], LlmInstance],
        rtol: float,
        atol: float,
    ):
        self.make_target_model = make_target_model
        self.make_reference_model = make_reference_model
        self.rtol = rtol
        self.atol = atol

    def run_and_assert_close(self, tokens: list[list[int]]):
        reference_prefill_results, reference_decode_results = self._run(
            self.make_reference_model, tokens=tokens
        )
        target_prefill_results, target_decode_results = self._run(
            self.make_target_model, tokens=tokens
        )

        self._assert_close(target_prefill_results, reference_prefill_results)
        self._assert_close(target_decode_results, reference_decode_results)

    def _run(
        self,
        make_model: Callable[[], LlmInstance],
        tokens: list[list[int]],
    ):
        instance = make_model()
        perplexity_eval = instance.make_perplexity_eval()
        prefill_results = perplexity_eval.prefill_cross_entropy(tokens)
        decode_results = perplexity_eval.decode_cross_entropy(tokens)
        assert all(result.valid for result in prefill_results)
        assert all(result.valid for result in decode_results)
        return prefill_results, decode_results

    def _assert_close(
        self,
        actual: list[LlmPerplexityEval.Result],
        expected: list[LlmPerplexityEval.Result],
    ):
        actual_scores = torch.tensor([r.score for r in actual], dtype=torch.float32)
        expected_scores = torch.tensor([r.score for r in expected], dtype=torch.float32)
        assert_tensor_close(
            actual_scores, expected_scores, rtol=self.rtol, atol=self.atol
        )


def run_perplexity_test_pipeline_parallel_eager_vs_eager(
    reference_theta: Theta,
    reference_config: LlamaModelConfig,
    tokens: list[list[int]],
    pipeline_parallelism_size: int = 2,
):
    """Check that pipeline-parallel Llm generates the same perplexity as its
    non-parallelized counterpart."""
    batch_size = len(tokens)
    device = reference_config.device

    reference_theta = reference_theta.transform(
        InferenceTensorTransforms.to_device(device)
    )
    reference_model = TorchInstance(
        reference_theta,
        reference_config,
        device=device,
        prefill_bs=batch_size,
        decode_bs=batch_size,
    )

    pp_config = deepcopy(reference_config)
    pp_config.parallelism_config = ParallelismConfig.default_config(
        block_count=reference_config.hp.block_count,
        pp=pipeline_parallelism_size,
    )
    pp_theta = Theta(reference_theta.flatten())
    pipeline_parallelize_llm_theta(pp_theta, pp_config.parallelism_config)

    pp_model = TorchInstance(
        pp_theta, pp_config, device=device, prefill_bs=batch_size, decode_bs=batch_size
    )

    def make_llm_instance(model: TorchInstance):
        page_sizes = llama_config_page_sizes(model.config)
        page_count = minimum_required_kv_cache_page_count_for_batch(
            tokens=tokens, config=model.config
        )

        return LlmInstance(
            model_instance=model,
            page_sizes=page_sizes,
            block_seq_stride=model.config.block_seq_stride,
            block_count=page_count,
            kv_cache_dtype=dtype_to_serialized_name(model.config.kv_cache_dtype),
            decode_topk_logits=None,
        )

    tester = LlmPerplexityCompare(
        make_target_model=functools.partial(make_llm_instance, model=pp_model),
        make_reference_model=functools.partial(
            make_llm_instance, model=reference_model
        ),
        atol=0,
        rtol=0,
    )
    tester.run_and_assert_close(tokens)


def clip_llm_block_count(
    theta: Theta, config: LlamaModelConfig, block_count: int
) -> tuple[Theta, LlamaModelConfig]:
    """Remove all trailing layers/blocks from the theta to align the desired block/layer count."""
    assert (
        config.pipeline_parallelism_size == 1 and config.tensor_parallelism_size == 1
    ), "Not supported"

    config = deepcopy(config)
    config.hp.block_count = block_count
    # Make sure block_count derivative values are recomputed.
    config_as_props = config.to_properties()
    del config_as_props["parallelism_config"]
    del config_as_props["tensor_parallelism_size"]
    del config_as_props["block_to_pipeline_map"]
    del config_as_props["pipeline_to_device_map"]
    config = LlamaModelConfig.from_properties(config_as_props)

    tree = theta.tree
    tree["blk"] = {k: v for k, v in tree["blk"].items() if int(k) < block_count}

    return Theta(tree), config
