# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import unittest
from pathlib import Path
import numpy as np

from sharktank.examples.paged_llm_v1 import *
from sharktank.utils import tokenizer
from sharktank.utils import hf_datasets
from sharktank.models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
from sharktank.layers import configs


class ExtendTest(unittest.TestCase):
    """
    Tests the extend functionality for the paged LLM model.

    This test validates that:
    1. Extend from token 0 produces the same results as prefill
    2. Extend from token 16 to 32 produces the same results as prefill from 0 to 32
    """

    def setUp(self):
        # Setup similar to prefill_tests.py but using a longer prompt for testing extend
        default_arguments = {
            "hf_dataset": "open_llama_3b_v2_f16_gguf",  # Using the proper dataset name from hf_datasets.py
            "tokenizer-config-json": Path("./open_llama_3b/tokenizer_config.json"),
            "prompt": [
                "Once upon a time, there was a little girl who loved to read books about dragons and wizards. One day, she found a mysterious book in the library."
            ],
            "device": None,
            "activation-dtype": "float32",
        }
        self.device = (
            torch.device(default_arguments["device"])
            if default_arguments["device"]
            else None
        )
        self.activation_dtype = getattr(torch, default_arguments["activation-dtype"])
        assert isinstance(self.activation_dtype, torch.dtype)

        # Make sure we test a model with an actual GGUF file
        test_model = default_arguments["hf_dataset"]
        self.data_files = hf_datasets.get_dataset(test_model).download(
            local_dir=Path(".")
        )

        # Ensure we have a GGUF file
        assert (
            "gguf" in self.data_files
        ), f"Model {test_model} must have GGUF files available"
        assert len(self.data_files["gguf"]) > 0, f"No GGUF files found for {test_model}"

        # Load the dataset
        self.dataset = Dataset.load(self.data_files["gguf"][0], file_type="gguf")

        # Get the tokenizer config from the dataset directory
        # OpenLLama uses a .model file and not tokenizer.json
        tokenizer_dir = self.data_files["tokenizer_config.json"][0].parent
        self.tokenizer_config = tokenizer.load_tokenizer(
            tokenizer_dir,
            tokenizer_type="transformers",
        )
        self.prompts = default_arguments["prompt"]
        self.block_seq_stride = 16  # Using a fixed stride for predictability

    def createConfigModel(self, kv_cache_type="paged"):
        """Create model config with specified KV cache type"""
        # For open_llama_3b, we'll use FP16 for better memory usage
        return LlamaModelConfig(
            hp=configs.LlamaHParams.from_gguf_props(self.dataset.properties),
            block_seq_stride=self.block_seq_stride,
            kv_cache_type=kv_cache_type,
            device=self.device,
            activation_dtype=torch.float16,  # Use FP16 for large model
            attention_dtype=torch.float16,  # Use FP16 for large model
            kv_cache_dtype=torch.float16,  # Use FP16 for large model
        )

    def run_prefill(self, model, prompt_tokens, seq_lens, cache_state=None):
        """Run standard prefill operation and return logits"""
        # Setup batch for prefill
        bs = prompt_tokens.shape[0]
        attention_mask = model.attention_mask(
            model.input_mask(seq_lens, prompt_tokens.shape[1])
        )

        # Setup cache and block IDs if not provided
        if cache_state is None:
            cache_state = model.cache.allocate(
                page_count=seq_lens.max().item() // model.cache.block_seq_stride + 1
            )

        # Create sequence block IDs
        seq_block_ids = []
        for seq_len in seq_lens:
            blocks_needed = int(np.ceil(seq_len.item() / self.block_seq_stride))
            row = list(
                range(1, blocks_needed + 1)
            )  # Start with 1 to match TorchGenerator
            seq_block_ids.append(row)

        # Pad sequence block IDs
        max_length = max(len(r) for r in seq_block_ids)
        padded_seq_block_ids = [r + (max_length - len(r)) * [0] for r in seq_block_ids]
        seq_block_ids_tensor = torch.tensor(padded_seq_block_ids, device=self.device)

        # Run prefill
        logits = model.prefill(
            prompt_tokens,
            attention_mask=attention_mask,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=cache_state,
        )

        return logits, cache_state

    def run_extend(self, model, prompt_tokens, seq_lens, start_positions, cache_state):
        """Run extend operation (prefill with start_positions) and return logits"""
        # Setup batch for extend
        bs = prompt_tokens.shape[0]
        attention_mask = model.attention_mask(
            model.input_mask(seq_lens, prompt_tokens.shape[1])
        )

        # Create sequence block IDs (same as for prefill)
        seq_block_ids = []
        for seq_len in seq_lens:
            blocks_needed = int(np.ceil(seq_len.item() / self.block_seq_stride))
            row = list(
                range(1, blocks_needed + 1)
            )  # Start with 1 to match TorchGenerator
            seq_block_ids.append(row)

        # Pad sequence block IDs
        max_length = max(len(r) for r in seq_block_ids)
        padded_seq_block_ids = [r + (max_length - len(r)) * [0] for r in seq_block_ids]
        seq_block_ids_tensor = torch.tensor(padded_seq_block_ids, device=self.device)

        # Run extend operation (prefill with start_positions)
        logits = model.prefill(
            prompt_tokens,
            attention_mask=attention_mask,
            start_positions=start_positions,
            seq_block_ids=seq_block_ids_tensor,
            cache_state=cache_state,
        )

        return logits

    def extract_tokens_from_logits(self, model, logits, seq_lens):
        """Extract token predictions from logits"""
        tokens = torch.tensor(
            model.extract_tokens_from_logits(logits, seq_lens)
        ).unsqueeze(1)
        return tokens

    def test_extend_from_zero_matches_prefill(self):
        """Test that extend from position 0 gives the same result as prefill"""
        # Create model with low precision to save memory
        config = self.createConfigModel(kv_cache_type="paged")
        model = PagedLlamaModelV1(self.dataset.root_theta, config)

        # Tokenize prompt but limit length to make it faster
        max_length = (
            16  # Make it divisible by block_seq_stride=16 to avoid index issues
        )
        token_ids, seq_lens = self.tokenizer_config.encode(
            self.prompts, pad_to_multiple_of=model.cache.pad_sequence_stride
        )

        # Tokenizer might return different formats - handle both cases
        if isinstance(token_ids, torch.Tensor):
            # Already a tensor, just limit length
            token_ids = token_ids[:, :max_length]
        elif isinstance(token_ids, list):
            # Convert to numpy and limit length
            token_ids = np.array(token_ids)[:, :max_length]

        # Limit sequence length to max_length
        seq_lens = torch.tensor([max_length], device=self.device)
        token_ids = torch.tensor(token_ids, device=self.device)

        # Run new prefill test that exactly fits the block size
        prefill_logits, cache_state = self.run_prefill(model, token_ids, seq_lens)
        prefill_tokens = self.extract_tokens_from_logits(
            model, prefill_logits, seq_lens
        )

        # Create new cache state for extend test
        # We need a new cache to avoid test interference
        cache_state_extend = model.cache.allocate(
            page_count=seq_lens.max().item() // model.cache.block_seq_stride + 1
        )

        # Run extend from position 0 (should be equivalent to prefill)
        start_positions = torch.zeros_like(seq_lens)
        extend_logits = self.run_extend(
            model, token_ids, seq_lens, start_positions, cache_state_extend
        )
        extend_tokens = self.extract_tokens_from_logits(model, extend_logits, seq_lens)

        # Check that logits and predicted tokens match
        # Use higher tolerance for float16
        rtol = 1e-2  # Higher tolerance for float16
        atol = 1e-2  # Higher tolerance for float16
        torch.testing.assert_close(prefill_logits, extend_logits, rtol=rtol, atol=atol)
        self.assertEqual(prefill_tokens.tolist(), extend_tokens.tolist())

    def test_extend_partial_matches_full_prefill(self):
        """Test that extend from position 16 produces the same final result as a full prefill"""
        # Create fixed exact prompt to test with
        config = self.createConfigModel(kv_cache_type="paged")
        model = PagedLlamaModelV1(self.dataset.root_theta, config)

        # Use a guaranteed working length for this test: exactly 32 tokens
        # which is 2 blocks of 16 tokens
        fixed_length = 32

        # Get tokens, ensure they're exactly the right length
        token_ids, seq_lens = self.tokenizer_config.encode(
            self.prompts, pad_to_multiple_of=16
        )

        # Tokenizer might return different formats - handle both cases
        if isinstance(token_ids, torch.Tensor):
            # Already a tensor
            token_ids = token_ids[:, :fixed_length]
        elif isinstance(token_ids, list):
            # Convert to numpy
            token_ids = np.array(token_ids)[:, :fixed_length]

        token_ids = torch.tensor(token_ids, device=self.device)

        # Use fixed lengths for clarity
        first_block_len = 16
        full_length = 32

        # Run prefill for first 16 tokens (first block)
        first_16_tokens = token_ids[:, :first_block_len]
        first_16_seq_lens = torch.tensor([first_block_len], device=self.device)

        # Create cache for first block prefill
        first_cache_state = model.cache.allocate(
            page_count=full_length // model.cache.block_seq_stride
        )

        # Run prefill for first block
        _, first_cache_state = self.run_prefill(
            model, first_16_tokens, first_16_seq_lens, cache_state=first_cache_state
        )

        # Now extend from token 16 to token 32
        full_tokens = token_ids[:, :full_length]
        full_seq_lens = torch.tensor([full_length], device=self.device)
        start_positions = torch.tensor([first_block_len], device=self.device)

        # Run extend using the first block cache
        extend_logits = self.run_extend(
            model, full_tokens, full_seq_lens, start_positions, first_cache_state
        )

        # Get final token prediction after extend
        extend_token_idx = torch.argmax(
            extend_logits[0, full_length - 1]
        )  # Last token position (0-indexed)
        extend_token_logit = extend_logits[0, full_length - 1, extend_token_idx]

        # Run full prefill from 0 to 32 with fresh cache
        full_cache_state = model.cache.allocate(
            page_count=full_length // model.cache.block_seq_stride
        )
        full_prefill_logits, _ = self.run_prefill(
            model, full_tokens, full_seq_lens, cache_state=full_cache_state
        )

        # Get final token prediction for full prefill
        full_prefill_token_idx = torch.argmax(full_prefill_logits[0, full_length - 1])
        full_prefill_token_logit = full_prefill_logits[
            0, full_length - 1, full_prefill_token_idx
        ]

        # Check that the final token prediction matches
        # Use a higher tolerance for float16
        rtol = 1e-2  # Higher tolerance for float16
        atol = 1e-2  # Higher tolerance for float16
        self.assertEqual(
            extend_token_idx.item(),
            full_prefill_token_idx.item(),
            "Predicted token index should match between extend and full prefill",
        )
        torch.testing.assert_close(
            extend_token_logit,
            full_prefill_token_logit,
            rtol=rtol,
            atol=atol,
            msg="Token logit values should match between extend and full prefill",
        )


if __name__ == "__main__":
    unittest.main()
