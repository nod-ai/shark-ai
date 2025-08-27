# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from sharktank.layers import (
    PagedLlamaAttentionBlock,
    PagedAttention,
    build_rotary_layer,
)
from sharktank.layers.testing import make_llama_attention_block_theta
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.quantizers import StaticScaledQuantizer, DynamicScaledQuantizer
from sharktank.utils.iree import oneshot_iree_run
from sharktank.utils.math import cosine_similarity


class PagedLlamaAttentionBlockTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.transformer_block_count = 13
        self.block_index = 1
        self.shard_count = 3
        self.head_count_kv = 2 * self.shard_count
        self.attention_head_count = 5 * self.head_count_kv
        self.attention_head_dim = (
            128  # Standard dimension that works with IREE GPU kernels
        )
        self.rms_epsilon = 0.01
        self.block_seq_stride = 17
        self.cache_partition_count = 2
        self.page_count = 23
        self.embedding_length = self.attention_head_count * self.attention_head_dim
        self.rope_dimension_count = self.attention_head_dim
        self.block_seqlen = 7
        self.max_seqlen = self.block_seq_stride * self.block_seqlen
        self.rope_freq_base = None
        self.batch_size = 3
        self.start_index = 0

    # @pytest.mark.xfail(
    #     torch.__version__ >= (2, 4),
    #     reason="https://github.com/nod-ai/shark-ai/issues/684",
    # )
    # @pytest.mark.skipif(
    #     torch.__version__ >= (2, 5),
    #     reason="https://github.com/nod-ai/shark-ai/issues/684, error slows down CI",
    # )
    def testExportNondecomposed(self):
        """Test float8 vs float32 attention block export and execution"""
        print("\n=== Float8 vs Float32 Attention Test ===")

        # CHANGE THIS TO TEST DIFFERENT FLOAT8 TYPES
        test_float8_dtype = (
            torch.float8_e4m3fnuz
        )  # Change to torch.float8_e4m3fn to test FN
        scale_factor = 2.0  # Use 1.0 for float8_e4m3fn

        # Create the base theta and input ONCE to ensure same weights for both tests
        torch.manual_seed(12345)  # Reset seed for reproducible weights
        base_theta = make_llama_attention_block_theta(
            block_idx=0,
            head_count=self.attention_head_count,
            head_count_kv=self.head_count_kv,
            head_dim=self.attention_head_dim,
            embedding_length=self.embedding_length,
        )

        # Same input for both tests
        torch.manual_seed(54321)  # Different seed for input
        h = torch.rand(
            [
                self.batch_size,
                self.max_seqlen,
                self.attention_head_count * self.attention_head_dim,
            ]
        )

        results = {}

        for test_name, use_float8 in [
            ("Float32 baseline", False),
            ("Float8 test", True),
        ]:
            print(f"\n--- {test_name} ---")

            dtype = test_float8_dtype if use_float8 else torch.float32

            cache = PagedAttention(
                transformer_block_count=self.transformer_block_count,
                attn_head_count=self.head_count_kv,
                attn_head_dim=self.attention_head_dim,
                cache_partition_count=self.cache_partition_count,
                block_seq_stride=self.block_seq_stride,
                cache_dtype=dtype,  # Keep this as dtype - critical for testing!
                attn_dtype=dtype,  # Keep this as dtype - critical for testing!
            )

            cache_state = cache.allocate(self.page_count)
            cache_state[0] = torch.rand(cache_state[0].shape, dtype=torch.float32).to(
                dtype
            )

            # Copy the base theta for this test
            import copy

            theta = copy.deepcopy(base_theta)

            # Quantize weights if testing float8
            if use_float8:
                # Use dynamic quantizer to avoid picking bad scales
                quantizer = DynamicScaledQuantizer(dtype=dtype)

                # Apply quantization to attention weights
                for key in ["attn_q", "attn_k", "attn_v", "attn_output"]:
                    if f"{key}.weight" in theta.tree:
                        original_weight = theta.tree[f"{key}.weight"]
                        theta.tree[f"{key}.weight"] = quantizer.quantize(
                            original_weight, name=f"{key}.weight"
                        )

            attn = PagedLlamaAttentionBlock(
                theta=theta,
                block_index=self.block_index,
                cache=cache,
                head_count=self.attention_head_count,
                head_dim=self.attention_head_dim,
                head_count_kv=self.head_count_kv,
                rms_epsilon=self.rms_epsilon,
                attention_kernel="torch",
                model_arch="llama",
            )

            seq_block_ids = torch.arange(self.batch_size * self.block_seqlen).view(
                self.batch_size, -1
            )

            embedding_module = build_rotary_layer(
                rope_dimension_count=self.rope_dimension_count,
                rope_freq_base=self.rope_freq_base,
            )

            class MyModule(torch.nn.Module):
                def forward(self, h, seq_block_ids, cache_state):
                    return attn.forward(
                        h,
                        seq_block_ids=seq_block_ids,
                        embedding=embedding_module,
                        cache_state=cache_state,
                    )

            mod = MyModule()

            # IREE test - this is the only way to run float8
            try:
                compile_args = [
                    "--iree-hal-target-device=hip",
                    "--iree-hip-target=gfx942",
                ]
                iree_result = oneshot_iree_run(
                    mod,
                    args=(h, seq_block_ids, cache_state),
                    device="hip",
                    compile_args=compile_args,
                )
                results[test_name] = iree_result[0]

                print(f"IREE run: SUCCESS")
                print(f"Output shape: {iree_result[0].shape}")
                print(
                    f"Output range: [{iree_result[0].min():.6f}, {iree_result[0].max():.6f}]"
                )

                if torch.isnan(iree_result[0]).any():
                    print(f"*** WARNING: NaN values detected! ***")
                if torch.isinf(iree_result[0]).any():
                    print(f"*** WARNING: Inf values detected! ***")

            except Exception as e:
                print(f"IREE run: FAILED - {type(e).__name__}: {e}")
                results[test_name] = None

        # Compare results
        if len(results) == 2 and all(r is not None for r in results.values()):
            float32_out = results["Float32 baseline"]
            float8_out = results["Float8 test"]
            print(float32_out, float8_out)
            mse = torch.mean((float32_out - float8_out) ** 2).item()
            max_diff = torch.max(torch.abs(float32_out - float8_out)).item()
            cos_sim = cosine_similarity(float32_out, float8_out).item()

            print(f"\n--- Comparison ---")
            print(f"MSE between float32 and float8: {mse:.8f}")
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"Cosine similarity: {cos_sim:.6f}")
        else:
            print(f"\n--- Comparison ---")
            print("Cannot compare - one or both runs failed")


if __name__ == "__main__":
    unittest.main()
