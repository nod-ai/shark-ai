# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GPT-OSS (OpenWeight) model tests.

This module contains comprehensive tests for both toy and actual GPT-OSS models:

1. Toy model tests (GptOssCrossEntropyTest):
   - Cross-entropy/perplexity calculation with generated small-scale models
   - Decode phase testing (testToyDecodePhase)

2. Actual weight tests (GptOssActualWeightsCrossEntropyTest):
   - Cross-entropy testing with real GPT-OSS weights
   - Decode phase testing (testActualWeightsDecodePhase)
   - Full generation testing with prefill + multiple decode steps (testActualWeightsPrefillDecodeGeneration)

3. IREE vs Eager tests (GptOssIreeVsEagerTest):
   - Compare IREE compiled vs PyTorch eager execution for both prefill and decode

The tests cover both prefill (initial prompt processing) and decode (token-by-token generation)
phases, mirroring real-world inference patterns.

To run actual weight tests, set the GPT_OSS_MODEL_PATH environment variable to point to
your GPT-OSS .irpa model file. Tests will be skipped if the path is not set or invalid.

Examples:
    # Set model path
    export GPT_OSS_MODEL_PATH=/path/to/gpt_oss_model.irpa

    # Run all actual weight tests (prefill + decode)
    pytest tests/models/gpt_oss/test_gpt_oss.py::GptOssActualWeightsCrossEntropyTest

    # Run specific decode test
    pytest tests/models/gpt_oss/test_gpt_oss.py::GptOssActualWeightsCrossEntropyTest::testActualWeightsDecodePhase

    # Run full generation test
    pytest tests/models/gpt_oss/test_gpt_oss.py::GptOssActualWeightsCrossEntropyTest::testActualWeightsPrefillDecodeGeneration
"""

import os
import unittest

import pytest
import torch

from sharktank.models.llm.llm import PagedLlmModelV1
from sharktank.models.gpt_oss.toy_gpt_oss import generate
from sharktank.types import Dataset
from sharktank.utils.export_artifacts import IreeCompileException
from sharktank.utils.testing import (
    is_mi300x,
    IreeVsEagerLLMTester,
    TempDirTestBase,
)


class GptOssCrossEntropyTest(unittest.TestCase):
    """Test GPT-OSS model perplexity calculation."""

    def testUnsharded(self):
        """Test perplexity calculation on toy GPT-OSS model."""
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)  # read irpa and replace it with that
        model = PagedLlmModelV1(theta=theta, config=config)

        # Use token sequence from your original openweight test
        ids = [3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]
        seq_len = len(ids)

        # Calculate blocks for paged attention
        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        ids = ids + [0] * padding

        ids = torch.asarray([ids], dtype=torch.int64)
        block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

        # Allocate KV cache
        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        # Run prefill
        logits = model.prefill(
            tokens=ids,
            seq_lens=torch.tensor([seq_len]),
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding
        ids = ids[:, :seq_len]
        logits = logits[:, :seq_len, :]

        # Calculate cross entropy (perplexity)
        ids = ids[0, 1:]
        logits = logits[0, :-1].to(torch.float32)
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

        # Expected value will need to be determined by running the test
        # For now, just check that it produces a reasonable value
        print(f"GPT-OSS Cross entropy: {cross_entropy.item()}")
        assert cross_entropy.item() > 0.0  # Should be positive
        assert cross_entropy.item() < 20.0  # Should be reasonable

        # TODO: Once we determine the expected value, replace with:
        # assert pytest.approx(expected_value, 1e-2) == cross_entropy

    def testToyDecodePhase(self):
        """Test decode phase with toy GPT-OSS model."""
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        # Initial token sequence for prefill
        initial_ids = [3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]
        seq_len = len(initial_ids)

        # === PREFILL PHASE ===
        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        prefill_ids = initial_ids + [0] * padding

        prefill_ids_tensor = torch.asarray([prefill_ids], dtype=torch.int64)
        block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

        # Allocate KV cache
        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        # Run prefill
        prefill_logits = model.prefill(
            tokens=prefill_ids_tensor,
            seq_lens=torch.tensor([seq_len]),
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding from prefill logits
        prefill_logits = prefill_logits[:, :seq_len, :]

        # Get next token from prefill
        next_token = torch.argmax(prefill_logits[0, -1, :], dim=-1).item()

        print(f"Toy GPT-OSS Prefill completed, next token: {next_token}")

        # === DECODE PHASE ===
        seq_lens = torch.tensor([seq_len])
        seq_lens = seq_lens + 1  # Increment for decode
        start_positions = [torch.tensor([seq_len])]  # Start position for decode

        # Create decode attention mask
        decode_attention_mask = model.decode_attention_mask(
            model.input_mask(
                seq_lens,
                block_ids.shape[1] * model.cache.block_seq_stride,
            ),
        )

        # Prepare next token tensor
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.int64)

        # Run decode
        decode_logits = model.decode(
            tokens=next_token_tensor,
            attention_mask=[decode_attention_mask],
            start_positions=start_positions,
            seq_block_ids=block_ids,
            cache_state=cache_state,
        )

        # Verify decode output
        assert decode_logits is not None
        assert decode_logits.shape[0] == 1  # Batch size
        assert decode_logits.shape[1] == 1  # Single token output
        assert decode_logits.shape[2] == config.hp.vocab_size  # Vocab size

        # Get next token from decode
        decode_next_token = torch.argmax(decode_logits[0, -1, :], dim=-1).item()

        print(f"Toy GPT-OSS Decode completed, next token: {decode_next_token}")

        # Basic sanity checks
        assert 0 <= decode_next_token < config.hp.vocab_size
        assert torch.isfinite(decode_logits).all()

        print("Toy GPT-OSS Decode phase test passed!")


def _create_config_from_dataset_properties(dataset):
    """Helper function to create LlamaModelConfig from dataset properties.

    Handles both GGUF-style and custom IRPA property formats.
    """
    from sharktank.layers.configs import LlamaModelConfig, LlamaHParams

    props = dataset.properties

    # Check if we have 'hparams' structure (from your IRPA creation script)
    if "hparams" in props:
        hparams = props["hparams"]
        # Direct mapping from your IRPA hparams structure
        return LlamaModelConfig(
            hp=LlamaHParams(
                model_arch=hparams.get("model_arch", "gpt-oss"),
                vocab_size=hparams.get("vocab_size", 201088),
                context_length=hparams.get("context_length", 4096),
                embedding_length=hparams.get("embedding_length", 2880),
                block_count=hparams.get("block_count", 24),
                feed_forward_length=hparams.get("feed_forward_length", 2880),
                attention_head_count=hparams.get("attention_head_count", 64),
                attn_head_dim=hparams.get("attn_head_dim", 64),
                attention_layer_norm_rms_epsilon=hparams.get(
                    "attention_layer_norm_rms_epsilon", 1e-5
                ),
                attention_head_count_kv=hparams.get("attention_head_count_kv", 8),
                rope_dimension_count=hparams.get("rope_dimension_count", 64),
                rope_freq_base=hparams.get("rope_freq_base", 150000.0),
                # MoE configuration
                expert_count=hparams.get("expert_count", 32),
                expert_used_count=hparams.get("expert_used_count", 4),
                expert_feed_forward_length=hparams.get("feed_forward_length", 2880),
                # GPT-OSS specific configs
                moe_block_type="PreGatherFFNMOE",
                use_moe_swiglu=True,
                sliding_window=hparams.get("sliding_window", 128),
                swiglu_limit=hparams.get("swiglu_limit", 7.0),
                rope_gpt_oss=True,
                use_fused_qkv=True,
                use_direct_expert_routing=True,
                use_residual_moe=True,
                use_ffn_norm=False,
                use_ffn_residual=False,
                moe_score_function="softmax",
                moe_activation_function="swiglu",
                normalize_moe_experts=False,
                # YaRN scaling
                yarn_factor=hparams.get("yarn_factor", 32.0),
                yarn_beta_slow=hparams.get("yarn_beta_slow", 1.0),
                yarn_beta_fast=hparams.get("yarn_beta_fast", 32.0),
                yarn_original_context_len=hparams.get(
                    "yarn_original_context_len", 4096
                ),
            ),
            block_seq_stride=16,  # Typical value for paged attention
            activation_dtype=torch.bfloat16,
            attention_dtype=torch.bfloat16,
            use_hf=False,
            dtype=torch.bfloat16,
        )
    else:
        # Fallback to direct property access for other formats
        return LlamaModelConfig(
            hp=LlamaHParams(
                model_arch="gpt-oss",
                vocab_size=props.get("vocab_size", 201088),
                context_length=props.get("context_length", 4096),
                embedding_length=props.get("embedding_length", 2880),
                block_count=props.get("block_count", 24),
                feed_forward_length=props.get("feed_forward_length", 2880),
                attention_head_count=props.get("attention_head_count", 64),
                attn_head_dim=props.get("attn_head_dim", 64),
                attention_layer_norm_rms_epsilon=props.get(
                    "attention_layer_norm_rms_epsilon", 1e-5
                ),
                attention_head_count_kv=props.get("attention_head_count_kv", 8),
                rope_dimension_count=props.get("rope_dimension_count", 64),
                rope_freq_base=props.get("rope_freq_base", 150000.0),
                # MoE configuration
                expert_count=props.get("expert_count", 32),
                expert_used_count=props.get("expert_used_count", 4),
                expert_feed_forward_length=props.get(
                    "expert_feed_forward_length", 2880
                ),
                # GPT-OSS specific configs
                moe_block_type="PreGatherFFNMOE",
                use_moe_swiglu=True,
                sliding_window=props.get("sliding_window", 128),
                swiglu_limit=props.get("swiglu_limit", 7.0),
                rope_gpt_oss=True,
                use_fused_qkv=True,
                use_direct_expert_routing=True,
                use_residual_moe=True,
                use_ffn_norm=False,
                use_ffn_residual=False,
                moe_score_function="softmax",
                moe_activation_function="swiglu",
                normalize_moe_experts=False,
                # YaRN scaling
                yarn_factor=props.get("yarn_factor", 32.0),
                yarn_beta_slow=props.get("yarn_beta_slow", 1.0),
                yarn_beta_fast=props.get("yarn_beta_fast", 32.0),
                yarn_original_context_len=props.get("yarn_original_context_len", 4096),
            ),
            block_seq_stride=16,  # Typical value for paged attention
            activation_dtype=torch.bfloat16,
            attention_dtype=torch.bfloat16,
            use_hf=False,
            dtype=torch.bfloat16,
        )


class GptOssActualWeightsCrossEntropyTest(unittest.TestCase):
    """Test GPT-OSS model perplexity calculation with actual weights."""

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("GPT_OSS_MODEL_PATH", "")),
        reason="GPT_OSS_MODEL_PATH environment variable not set or file doesn't exist",
    )
    def testInspectDatasetProperties(self):
        """Debug helper to inspect the actual dataset properties."""
        model_path = os.getenv("GPT_OSS_MODEL_PATH")
        if not model_path:
            self.skipTest("GPT_OSS_MODEL_PATH environment variable not set")

        # Load actual model weights
        dataset = Dataset.load(model_path)

        print("=== Dataset Properties Structure ===")
        print("Keys:", list(dataset.properties.keys()))
        for key, value in dataset.properties.items():
            if isinstance(value, dict):
                print(f"{key}: (dict with keys: {list(value.keys())})")
                if key == "hparams":
                    print("  hparams contents:")
                    for hkey, hvalue in value.items():
                        print(f"    {hkey}: {hvalue}")
            else:
                print(f"{key}: {value}")

        print("\n=== Theta Structure ===")
        print("Theta keys (first 10):", list(dataset.root_theta.keys())[:10])

        # This test always passes - it's just for inspection
        assert True

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("GPT_OSS_MODEL_PATH", "")),
        reason="GPT_OSS_MODEL_PATH environment variable not set or file doesn't exist",
    )
    def testActualWeights(self):
        """Test perplexity calculation on actual GPT-OSS model weights."""
        model_path = os.getenv("GPT_OSS_MODEL_PATH")
        if not model_path:
            self.skipTest("GPT_OSS_MODEL_PATH environment variable not set")

        # torch.set_default_dtype(torch.float32)

        # Load actual model weights
        dataset = Dataset.load(model_path)
        theta = dataset.root_theta

        # Create config from dataset properties
        config = _create_config_from_dataset_properties(dataset)

        model = PagedLlmModelV1(theta=theta, config=config)

        # Use a representative token sequence
        ids = [3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]
        seq_len = len(ids)

        # Calculate blocks for paged attention
        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        ids = ids + [0] * padding

        ids = torch.asarray([ids], dtype=torch.int64)
        block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

        # Allocate KV cache
        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        # Run prefill
        logits = model.prefill(
            tokens=ids,
            seq_lens=torch.tensor([seq_len]),
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding
        ids = ids[:, :seq_len]
        logits = logits[:, :seq_len, :]

        # Calculate cross entropy (perplexity)
        ids = ids[0, 1:]
        logits = logits[0, :-1].to(torch.float32)
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

        print(f"GPT-OSS Actual Weights Cross entropy: {cross_entropy.item()}")
        assert cross_entropy.item() > 0.0  # Should be positive
        assert cross_entropy.item() < 50.0  # Should be reasonable for actual model

        # TODO: Once we determine the expected value from actual GPT-OSS, replace with:
        # assert pytest.approx(expected_value, 1e-2) == cross_entropy

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("GPT_OSS_MODEL_PATH", "")),
        reason="GPT_OSS_MODEL_PATH environment variable not set or file doesn't exist",
    )
    def testActualWeightsDecodePhase(self):
        """Test decode phase with actual GPT-OSS model weights."""
        model_path = os.getenv("GPT_OSS_MODEL_PATH")
        if not model_path:
            self.skipTest("GPT_OSS_MODEL_PATH environment variable not set")

        torch.set_default_dtype(torch.float32)

        # Load actual model weights
        dataset = Dataset.load(model_path)
        theta = dataset.root_theta

        # Create config from dataset properties
        config = _create_config_from_dataset_properties(dataset)

        model = PagedLlmModelV1(theta=theta, config=config)

        # Initial token sequence for prefill
        initial_ids = [3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]
        seq_len = len(initial_ids)

        # === PREFILL PHASE ===
        # Calculate blocks for paged attention
        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        prefill_ids = initial_ids + [0] * padding

        prefill_ids_tensor = torch.asarray([prefill_ids], dtype=torch.int64)
        block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

        # Allocate KV cache
        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        # Run prefill
        prefill_logits = model.prefill(
            tokens=prefill_ids_tensor,
            seq_lens=torch.tensor([seq_len]),
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding from prefill logits
        prefill_logits = prefill_logits[:, :seq_len, :]

        # Get next token from prefill
        next_token = torch.argmax(prefill_logits[0, -1, :], dim=-1).item()

        print(f"GPT-OSS Prefill completed, next token: {next_token}")

        # === DECODE PHASE ===
        seq_lens = torch.tensor([seq_len])
        seq_lens = seq_lens + 1  # Increment for decode
        start_positions = [torch.tensor([seq_len])]  # Start position for decode

        # Create decode attention mask
        decode_attention_mask = model.decode_attention_mask(
            model.input_mask(
                seq_lens,
                block_ids.shape[1] * model.cache.block_seq_stride,
            ),
        )

        # Prepare next token tensor
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.int64)

        # Run decode
        decode_logits = model.decode(
            tokens=next_token_tensor,
            attention_mask=[decode_attention_mask],
            start_positions=start_positions,
            seq_block_ids=block_ids,
            cache_state=cache_state,
        )

        # Verify decode output
        assert decode_logits is not None
        assert decode_logits.shape[0] == 1  # Batch size
        assert decode_logits.shape[1] == 1  # Single token output
        assert decode_logits.shape[2] == config.hp.vocab_size  # Vocab size

        # Get next token from decode
        decode_next_token = torch.argmax(decode_logits[0, -1, :], dim=-1).item()

        print(f"GPT-OSS Decode completed, next token: {decode_next_token}")

        # Basic sanity checks
        assert 0 <= decode_next_token < config.hp.vocab_size
        assert torch.isfinite(decode_logits).all()

        print("GPT-OSS Decode phase test passed!")

    @pytest.mark.skipif(
        not os.path.exists(os.getenv("GPT_OSS_MODEL_PATH", "")),
        reason="GPT_OSS_MODEL_PATH environment variable not set or file doesn't exist",
    )
    def testActualWeightsPrefillDecodeGeneration(self):
        """Test full generation sequence (prefill + multiple decode steps) with actual GPT-OSS weights."""
        model_path = os.getenv("GPT_OSS_MODEL_PATH")
        if not model_path:
            self.skipTest("GPT_OSS_MODEL_PATH environment variable not set")

        torch.set_default_dtype(torch.float32)

        # Load actual model weights
        dataset = Dataset.load(model_path)
        theta = dataset.root_theta

        # Create config from dataset properties
        config = _create_config_from_dataset_properties(dataset)

        model = PagedLlmModelV1(theta=theta, config=config)

        # Initial token sequence
        initial_ids = [3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]
        generated_tokens = []
        max_new_tokens = 3  # Generate 3 new tokens for testing

        seq_len = len(initial_ids)
        current_seq_len = seq_len

        # === PREFILL PHASE ===
        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        prefill_ids = initial_ids + [0] * padding

        prefill_ids_tensor = torch.asarray([prefill_ids], dtype=torch.int64)
        block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

        # Allocate KV cache
        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        # Run prefill
        logits = model.prefill(
            tokens=prefill_ids_tensor,
            seq_lens=torch.tensor([seq_len]),
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding and get first token
        logits = logits[:, :seq_len, :]
        next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
        generated_tokens.append(next_token)

        print(f"GPT-OSS Prefill completed. First generated token: {next_token}")

        # === DECODE PHASE (Multiple steps) ===
        for step in range(
            max_new_tokens - 1
        ):  # -1 because we got first token from prefill
            current_seq_len += 1
            seq_lens = torch.tensor([current_seq_len])
            start_positions = [torch.tensor([current_seq_len - 1])]

            # Create decode attention mask
            decode_attention_mask = model.decode_attention_mask(
                model.input_mask(
                    seq_lens,
                    block_ids.shape[1] * model.cache.block_seq_stride,
                ),
            )

            # Prepare current token tensor
            current_token_tensor = torch.tensor([[next_token]], dtype=torch.int64)

            # Run decode
            decode_logits = model.decode(
                tokens=current_token_tensor,
                attention_mask=[decode_attention_mask],
                start_positions=start_positions,
                seq_block_ids=block_ids,
                cache_state=cache_state,
            )

            # Get next token
            next_token = torch.argmax(decode_logits[0, -1, :], dim=-1).item()
            generated_tokens.append(next_token)

            print(f"GPT-OSS Decode step {step + 1}, generated token: {next_token}")

        # Verify generation
        assert len(generated_tokens) == max_new_tokens
        for token in generated_tokens:
            assert 0 <= token < config.hp.vocab_size

        print(
            f"GPT-OSS Full generation test passed! Generated tokens: {generated_tokens}"
        )

        # Basic coherence check - tokens should be valid
        assert all(isinstance(token, int) for token in generated_tokens)
        assert all(0 <= token < config.hp.vocab_size for token in generated_tokens)


@pytest.mark.usefixtures("iree_flags", "device")
@is_mi300x
class GptOssIreeVsEagerTest(TempDirTestBase):
    """Test GPT-OSS IREE vs Eager execution."""

    @pytest.mark.xfail(
        raises=IreeCompileException,
        reason="GPT-OSS IREE compilation may need additional work",
    )
    def testUnshardedToyIreeVsEager(self):
        """Test IREE vs Eager execution for toy GPT-OSS model."""
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
        # Test both prefill and decode phases
        tester.run_and_compare_iree_vs_eager()

    @pytest.mark.xfail(
        raises=IreeCompileException,
        reason="GPT-OSS IREE compilation may need additional work",
    )
    @pytest.mark.skipif(
        not os.path.exists(os.getenv("GPT_OSS_MODEL_PATH", "")),
        reason="GPT_OSS_MODEL_PATH environment variable not set or file doesn't exist",
    )
    def testActualWeightsIreeVsEager(self):
        """Test IREE vs Eager execution for actual GPT-OSS model weights."""
        model_path = os.getenv("GPT_OSS_MODEL_PATH")
        if not model_path:
            pytest.skip("GPT_OSS_MODEL_PATH environment variable not set")

        # Load actual model weights
        dataset = Dataset.load(model_path)
        theta = dataset.root_theta

        # Create config from dataset properties
        config = _create_config_from_dataset_properties(dataset)

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
