# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch

from copy import deepcopy, copy
from .llm import PagedLlmModelV1
from sharktank.types import InferenceTensorTransforms, Theta
from sharktank.types.pipelining import pipeline_parallelize_llm_theta
from sharktank.layers import CacheAllocation, LlamaModelConfig, ParallelismConfig
from sharktank.utils.attention import *
from sharktank.utils.math import round_up_to_multiple_of
from sharktank.utils.llm_utils import (
    LlmPerplexityEval,
    LlmInstance,
    TorchInstance,
    llama_config_page_sizes,
    minimum_required_kv_cache_page_count_for_batch,
)
from sharktank.utils.testing import assert_cosine_similarity_close, assert_tensor_close
from typing import Any, Callable, Tuple, OrderedDict


def make_random_kv_cache_state(
    model: PagedLlmModelV1, page_count: int
) -> CacheAllocation:
    cache_state = model.cache.allocate(page_count=page_count)
    cache_state.allocation = [
        torch.rand_like(tensor, dtype=torch.float32).to(dtype=tensor.dtype)
        for tensor in cache_state.allocation
    ]
    return cache_state


def make_random_decode_args(
    model: PagedLlmModelV1, batch_size: int
) -> OrderedDict[str, Any]:
    prefill_seq_lens = torch.randint(
        size=[batch_size],
        low=1,
        high=min(
            2 * model.config.block_seq_stride,
            model.config.hp.context_length,
        )
        - 1,
        dtype=torch.int64,
        device=model.device,
    )

    start_positions = prefill_seq_lens
    seq_lens = prefill_seq_lens + 1
    batch_seq_len = round_up_to_multiple_of(
        int(torch.max(seq_lens)), model.config.block_seq_stride
    )
    decode_token_ids = torch.randint(
        low=0,
        high=model.config.hp.vocab_size,
        size=[batch_size, 1],
        dtype=torch.int64,
    )
    seq_block_ids = torch.arange(
        batch_size * batch_seq_len // model.config.block_seq_stride,
        device=model.device,
    ).view(batch_size, -1)
    cache_state = make_random_kv_cache_state(
        model=model, page_count=seq_block_ids.numel() + batch_size
    )
    return OrderedDict(
        [
            ("tokens", decode_token_ids),
            ("seq_lens", seq_lens),
            ("start_positions", start_positions),
            ("seq_block_ids", seq_block_ids),
            ("cache_state", cache_state),
        ]
    )


def make_random_prefill_args(
    model: PagedLlmModelV1, batch_size: int
) -> OrderedDict[str, Any]:
    seq_lens = torch.randint(
        size=[batch_size],
        low=1,
        high=min(
            2 * model.config.block_seq_stride,
            model.config.hp.context_length,
        )
        - 1,
        dtype=torch.int64,
        device=model.device,
    )
    batch_seq_len = round_up_to_multiple_of(
        int(torch.max(seq_lens)), model.config.block_seq_stride
    )
    token_ids = torch.randint(
        low=0,
        high=model.config.hp.vocab_size,
        size=[batch_size, batch_seq_len],
        dtype=torch.int64,
        device=model.device,
    )

    seq_block_ids = torch.arange(
        batch_size * batch_seq_len // model.config.block_seq_stride,
        device=model.device,
    ).view(batch_size, -1)
    cache_state = make_random_kv_cache_state(
        model=model, page_count=seq_block_ids.numel() + batch_size
    )
    return OrderedDict(
        [
            ("tokens", token_ids),
            ("seq_lens", seq_lens),
            ("seq_block_ids", seq_block_ids),
            ("cache_state", cache_state),
        ]
    )


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


def convert_llm_kwargs(
    reference_kwargs: dict[str, Any],
    target_model: PagedLlmModelV1,
    reference_model: PagedLlmModelV1,
) -> dict[str, Any]:
    """Derive target model args from reference model args."""
    target_config = target_model.config
    reference_config = reference_model.config
    assert target_config.block_seq_stride == reference_config.block_seq_stride
    assert reference_config.pipeline_parallelism_size == 1
    assert reference_config.tensor_parallelism_size == 1
    assert target_config.tensor_parallelism_size == 1

    target_kwargs = {
        k: v.to(device=target_config.device)
        for k, v in reference_kwargs.items()
        if k != "cache_state"
    }

    reference_cache_state: CacheAllocation = reference_kwargs["cache_state"]
    assert len(reference_cache_state.allocation) == 1
    target_cache_state_tensor_list = [
        reference_cache_state.allocation[0].to(
            dtype=target_config.kv_cache_dtype, device=target_config.device
        )
    ]

    if target_config.pipeline_parallelism_size != 1:
        page_count = reference_cache_state.allocation[0].shape[0]
        target_cache_state = target_model.cache.allocate(page_count)
        cache_slab_dim = 1
        target_cache_state_sizes = [
            t.shape[cache_slab_dim] for t in target_cache_state.allocation
        ]
        target_cache_state_tensor_list = target_cache_state_tensor_list[0].split(
            target_cache_state_sizes, dim=cache_slab_dim
        )

    target_cache_numel = sum(t.numel() for t in target_cache_state_tensor_list)
    assert target_cache_numel == reference_cache_state.allocation[0].numel()
    target_kwargs["cache_state"] = CacheAllocation(target_cache_state_tensor_list)


class EagerVsEagerLLMTester:
    def __init__(
        self,
        target_model: PagedLlmModelV1,
        reference_model: PagedLlmModelV1,
        batch_size: int = 1,
    ):
        self.target_model = target_model
        self.reference_model = reference_model
        self.reference_prefill_kwargs = make_random_prefill_args(
            reference_model, batch_size
        )
        self.reference_decode_kwargs = make_random_decode_args(
            reference_model, batch_size
        )
        self.target_prefill_kwargs = convert_llm_kwargs(
            self.reference_prefill_kwargs,
            target_model=self.target_model,
            reference_model=self.reference_model,
        )
        self.target_decode_kwargs = convert_llm_kwargs(
            self.reference_decode_kwargs,
            target_model=self.target_model,
            reference_model=self.reference_model,
        )

    def run_and_compare(self):
        self.run_reference_model()
        self.run_target_model()
        self.assert_results_close()

    def run_reference_model(self):
        self.reference_prefill_results = self.reference_model.prefill(
            **self.reference_prefill_kwargs
        )
        self.reference_cache_state_post_prefill = deepcopy(
            self.reference_prefill_kwargs["cache_state"]
        )
        self.reference_decode_results = self.reference_model.decode(
            **self.reference_decode_kwargs
        )
        self.reference_cache_state_post_decode = deepcopy(
            self.reference_decode_kwargs["cache_state"]
        )

    def run_target_model(self):
        self.target_prefill_results = self.target_model.prefill(
            **self.target_prefill_kwargs
        )
        self.target_cache_state_post_prefill = deepcopy(
            self.target_prefill_kwargs["cache_state"]
        )
        self.target_decode_results = self.target_model.decode(
            **self.target_decode_kwargs
        )
        self.target_cache_state_post_decode = deepcopy(
            self.target_decode_kwargs["cache_state"]
        )

    def assert_results_close(self):
        assert_cosine_similarity_close(
            actual=self.target_prefill_results,
            expected=self.reference_prefill_results,
            dim=-1,
            atol=1e-3,
        )
        assert_cosine_similarity_close(
            actual=self.target_cache_state_post_prefill,
            expected=self.reference_cache_state_post_prefill,
            atol=1e-3,
        )

        assert_cosine_similarity_close(
            actual=self.target_decode_results,
            expected=self.reference_decode_results,
            dim=-1,
            atol=1e-3,
        )
        assert_cosine_similarity_close(
            actual=self.target_cache_state_post_decode,
            expected=self.reference_cache_state_post_decode,
            atol=1e-3,
        )


class LlmPerplexityCompare:
    """Compare the perplexity of one implementation against another.
    Checks that the 2 models' results do not deviate for each other."""

    def __init__(
        self,
        make_target_model: Callable[[], TorchInstance],
        make_reference_model: Callable[[], TorchInstance],
    ):
        self.make_target_model = make_target_model
        self.make_reference_model = make_reference_model

    def run_and_assert_close(self, tokens: list[list[int]]):
        reference_prefill_results, reference_decode_results = self._run(
            self.make_reference_model, tokens=tokens
        )
        target_prefill_results, target_decode_results = self._run(
            self.make_target_model, tokens=tokens
        )

        self._assert_close(target_prefill_results, reference_prefill_results)
        self._assert_close(target_decode_results, reference_decode_results)

    def _run(self, make_model: Callable[[], TorchInstance], tokens: list[list[int]]):
        model = make_model()
        page_sizes = llama_config_page_sizes(model.config)
        page_count = minimum_required_kv_cache_page_count_for_batch(
            tokens=tokens, config=model.config
        )

        instance = LlmInstance(
            model_instance=model,
            page_sizes=page_sizes,
            block_seq_stride=model.config.block_seq_stride,
            block_count=page_count,
            decode_topk_logits=None,
        )
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
        assert_tensor_close(actual_scores, expected_scores, atol=1e-2, rtol=1e-2)


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

    tester = LlmPerplexityCompare(
        make_target_model=lambda: pp_model, make_reference_model=lambda: reference_model
    )
    tester.run_and_assert_close(tokens)


def clip_llm_layers(
    theta: Theta, config: LlamaModelConfig, block_count: int
) -> tuple[Theta, LlamaModelConfig]:
    """Remove all trailing layers/blocks from the theta to align the desired block/layer count."""
    assert (
        config.pipeline_parallelism_size == 1 and config.tensor_parallelism_size == 1
    ), "Not supported"

    config = deepcopy(config)
    config.hp.block_count
    # Make sure block_count derivative values are recomputed.
    config.parallelism_config = None
    config.__post_init__()

    tree = theta.tree
    tree["blk"] = {k: v for k, v in tree["blk"].items() if int(k) < block_count}

    return Theta(tree), config
