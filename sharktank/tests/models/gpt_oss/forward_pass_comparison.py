"""Forward pass comparison tests between sharktank and reference GPT-OSS implementations."""

import unittest
import torch
import torch.nn.functional as F
import logging

from sharktank.models.gpt_oss.toy_gpt_oss import (
    generate_analytical,
    copy_weights_to_reference,
)
from sharktank.models.gpt_oss.orig_pytorch_model import (
    ModelConfig,
    Transformer,
    RotaryEmbedding,
    sdpa,
)
from sharktank.utils.llm_utils import TorchInstance

from sharktank.layers.paged_attention import PagedGQAttention, build_cache
import math


class ForwardPassComparisonTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        torch.set_default_dtype(torch.bfloat16)
        self.seed = 12345
        torch.manual_seed(self.seed)
        self.test_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]
        self.expected_pattern = [0.0, 1.0, -1.0, 0.5, 2.0]
        self.input_tokens = torch.tensor([[0, 1, 2]], dtype=torch.long)
        self.initialized_model()

    def initialized_model(self):
        # Initialize sharktank model
        self.shark_theta, self.shark_config = generate_analytical(self.seed)
        self.hp = self.shark_config.hp
        self.shark_model = TorchInstance(
            theta=self.shark_theta, config=self.shark_config
        )._model
        self.shark_model.eval()
        # Initialize reference model
        ref_config = ModelConfig(
            num_hidden_layers=self.hp.block_count,
            num_experts=self.hp.expert_count,
            experts_per_token=self.hp.expert_used_count,
            vocab_size=self.hp.vocab_size,
            hidden_size=self.hp.embedding_length,
            intermediate_size=self.hp.feed_forward_length,
            head_dim=self.hp.attn_head_dim,
            num_attention_heads=self.hp.attention_head_count,
            num_key_value_heads=self.hp.attention_head_count_kv,
            sliding_window=self.hp.sliding_window if self.hp.sliding_window else 128,
            initial_context_length=self.hp.context_length,
            rope_theta=self.hp.rope_freq_base,
            rope_scaling_factor=1.0,  # Disable YARN for tiny head_dim compatibility
        )

        self.ref_model = Transformer(ref_config, device=torch.device("cpu"))
        self.ref_model.eval()

        self.ref_model.vocab_size = ref_config.vocab_size
        self.ref_model.hidden_size = ref_config.hidden_size
        self.ref_model.num_hidden_layers = ref_config.num_hidden_layers

        copy_weights_to_reference(self.shark_theta, self.ref_model, self.hp)

    def test_token_embeddings(self):
        """Test token embeddings match expected analytical pattern."""
        with torch.no_grad():
            shark_emb = self.shark_model.token_embedding(self.input_tokens)
            ref_emb = self.ref_model.embedding(self.input_tokens)

        expected_embeddings = {
            0: [0.0, 1.0, -1.0, 0.5],
            1: [2.0, 0.0, 1.0, -1.0],
            2: [0.5, 2.0, 0.0, 1.0],
        }

        for token_idx in range(3):
            shark_values = shark_emb[0, token_idx, :].tolist()
            ref_values = ref_emb[0, token_idx, :].tolist()
            expected = expected_embeddings[token_idx]

            for i, (actual, exp) in enumerate(zip(shark_values, expected)):
                self.assertAlmostEqual(
                    actual,
                    exp,
                    places=3,
                    msg=f"Token {token_idx} position {i} mismatch",
                )

            match = torch.allclose(
                shark_emb[0, token_idx, :],
                ref_emb[0, token_idx, :],
                rtol=1e-4,
                atol=1e-4,
            )
            self.assertTrue(match)

            self.logger.debug(
                f"Token {token_idx} - Sharktank: {shark_values}, Reference: {ref_values}, Expected: {expected}"
            )

    def test_rmsnorm(self):
        """Test RMSNorm computation."""
        attn_norm_weight = self.shark_theta("blk", 0, "attn_norm", "weight").as_torch()
        shark_emb = self.shark_model.token_embedding(self.input_tokens)
        x = shark_emb[0, 0, :]  # Token 0: [0.0, 1.0, -1.0, 0.5]

        # Manual RMSNorm calculation
        x_float = x.float()
        mean_sq = torch.mean(x_float**2)
        eps = 1e-5
        rms = torch.sqrt(mean_sq + eps)
        normalized = x_float / rms
        normed_scaled = normalized * attn_norm_weight.float()
        with torch.no_grad():
            shark_block = self.shark_model.attn_blocks[0]
            shark_norm = shark_block.attn.attn_norm(shark_emb)
            shark_result = shark_norm[0, 0, :]

            ref_emb = self.ref_model.embedding(self.input_tokens)
            ref_block = self.ref_model.block[0]
            ref_norm = ref_block.attn.norm(ref_emb)
            ref_result = ref_norm[0, 0, :]

            torch.testing.assert_close(
                shark_result.float(), normed_scaled, rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                ref_result.float(), normed_scaled, rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                shark_result.float(), ref_result.float(), rtol=1e-4, atol=1e-4
            )

        self.logger.debug(
            f"RMSNorm - Sharktank: {shark_result.tolist()}, Reference: {ref_result.tolist()}, Hand: {normed_scaled.tolist()}"
        )

    def test_qkv_projection(self):
        """Test fused QKV projection."""
        qkv_weight = self.shark_theta("blk", 0, "attn", "wqkv", "weight").as_torch()
        qkv_bias = self.shark_theta("blk", 0, "attn", "wqkv", "bias").as_torch()

        with torch.no_grad():
            shark_emb = self.shark_model.token_embedding(self.input_tokens)
            shark_block = self.shark_model.attn_blocks[0]
            h_norm = shark_block.attn.attn_norm(shark_emb)[0, 0, :]

        # Manual QKV calculation
        hand_qkv_output = torch.matmul(h_norm, qkv_weight.t()) + qkv_bias
        q_size = self.hp.attention_head_count * self.hp.attn_head_dim
        kv_size = self.hp.attention_head_count_kv * self.hp.attn_head_dim

        hand_q = hand_qkv_output[:q_size]
        hand_k = hand_qkv_output[q_size : q_size + kv_size]
        hand_v = hand_qkv_output[q_size + kv_size : q_size + 2 * kv_size]

        with torch.no_grad():
            shark_qkv = shark_block.attn.attn_qkv(h_norm)
            shark_q = shark_qkv[:q_size]
            shark_k = shark_qkv[q_size : q_size + kv_size]
            shark_v = shark_qkv[q_size + kv_size : q_size + 2 * kv_size]

            ref_emb = self.ref_model.embedding(self.input_tokens)
            ref_block = self.ref_model.block[0]
            ref_norm = ref_block.attn.norm(ref_emb)[0, 0, :]
            ref_qkv = ref_block.attn.qkv(ref_norm)

            ref_q = ref_qkv[:q_size]
            ref_k = ref_qkv[q_size : q_size + kv_size]
            ref_v = ref_qkv[q_size + kv_size : q_size + 2 * kv_size]
        torch.testing.assert_close(shark_q, hand_q, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(shark_k, hand_k, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(shark_v, hand_v, rtol=1e-4, atol=1e-4)

        torch.testing.assert_close(ref_q, hand_q, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ref_k, hand_k, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(ref_v, hand_v, rtol=1e-4, atol=1e-4)

        self.logger.debug(
            f"QKV - Q: Hand={hand_q.tolist()}, Shark={shark_q.tolist()}, Ref={ref_q.tolist()}"
        )

    def test_rotary_embedding(self):
        """Test RoPE implementation with simple 0/1 values."""
        bs = 1
        seq_len = 3
        n_heads = self.hp.attention_head_count
        head_dim = self.hp.attn_head_dim

        q_ref = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],  # token 0: [1, 0] for both heads
                [[0.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            dtype=torch.bfloat16,
        )

        k_ref = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            dtype=torch.bfloat16,
        )

        q_shark = q_ref.unsqueeze(0)
        k_shark = k_ref.unsqueeze(0)
        position_ids = torch.arange(0, seq_len, device=q_shark.device)[None, :].repeat(
            bs, 1
        )

        # YARN disabled (factor=1.0) to avoid assertion with head_dim=2
        with torch.no_grad():
            from sharktank.layers.rotary_embedding_hf import RotaryEmbeddingLayer

            shark_rope = RotaryEmbeddingLayer(
                head_dim=head_dim,
                rope_theta=self.hp.rope_freq_base,
                use_base_frequency_scaling=False,
                interleaved=False,
                yarn_beta_slow=self.hp.yarn_beta_slow,
                yarn_beta_fast=self.hp.yarn_beta_fast,
                yarn_factor=1.0,
                yarn_original_context_len=self.hp.yarn_original_context_len,
            )
            cossin_cache = shark_rope.compute_sincos_cache(position_ids, q_shark.dtype)
            shark_q = shark_rope(q_shark, cossin_cache)
            shark_k = shark_rope(k_shark, cossin_cache)

            ref_rope = RotaryEmbedding(
                head_dim=head_dim,
                base=int(self.hp.rope_freq_base),
                dtype=torch.bfloat16,
                initial_context_length=self.hp.yarn_original_context_len,
                scaling_factor=1.0,
                ntk_alpha=self.hp.yarn_beta_slow,
                ntk_beta=self.hp.yarn_beta_fast,
            )
            ref_q, ref_k = ref_rope(q_ref, k_ref)
        torch.testing.assert_close(shark_q.squeeze(0), ref_q, rtol=2e-2, atol=1e-2)
        torch.testing.assert_close(shark_k.squeeze(0), ref_k, rtol=2e-2, atol=1e-2)

        self.logger.debug(
            f"RoPE - Shark Q[0,0,0,:]: {shark_q[0, 0, 0, :].tolist()}, Ref Q[0,0,:]: {ref_q[0, 0, :].tolist()}"
        )

    def test_sdpa_vs_paged_attention_prefill(self):
        """Compare reference sdpa with sharktank paged attention."""

        seq_len = 6
        bs = 1
        n_kv_heads = self.hp.attention_head_count_kv
        n_heads = self.hp.attention_head_count
        head_dim = self.hp.attn_head_dim
        q_mult = n_heads // n_kv_heads
        dtype = torch.bfloat16

        # Deterministic test tensors
        q_ref = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 1.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 1.0]]],
            ],
            dtype=dtype,
        )

        k_ref = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ],
            dtype=dtype,
        )

        v_ref = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ],
            dtype=dtype,
        )

        ref_sinks = self.ref_model.block[0].attn.sinks
        sm_scale = 1.0 / (head_dim**0.5)
        sliding_window = self.hp.sliding_window if self.hp.sliding_window else 0

        with torch.no_grad():
            # Reference implementation
            ref_out = sdpa(
                q_ref,
                k_ref,
                v_ref,
                S=ref_sinks,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
            )
            ref_out = ref_out.view(seq_len, n_kv_heads, q_mult, head_dim)

            # Paged attention setup
            kv_cache = build_cache(
                transformer_block_count=1,
                attn_head_count=n_kv_heads,
                attn_head_dim=head_dim,
                block_seq_stride=seq_len,
                cache_dtype=dtype,
            )
            pa = PagedGQAttention(
                kv_cache=kv_cache,
                transformer_block_index=0,
                attn_dtype=dtype,
                activation_dtype=dtype,
                use_rope=False,
                attention_chunk_size=None,
            )

            # Convert to paged attention format
            q_pa = q_ref.unsqueeze(0).reshape(bs, seq_len, n_heads, head_dim)
            k_pa = k_ref.unsqueeze(0).reshape(bs, seq_len, n_kv_heads, head_dim)
            v_pa = v_ref.unsqueeze(0).reshape(bs, seq_len, n_kv_heads, head_dim)

            # Cache setup
            blocks = math.ceil(seq_len / kv_cache.block_seq_stride)
            seq_block_ids = torch.arange(blocks, dtype=torch.int64).unsqueeze(0)
            cache_state = pa.allocate(page_count=blocks)
            seq_lens = torch.tensor([seq_len], dtype=torch.long)

            # Paged attention forward
            pa_out = pa.forward_prefill(
                q=q_pa,
                k=k_pa,
                v=v_pa,
                cache_state=cache_state,
                seq_lens=seq_lens,
                seq_block_ids=seq_block_ids,
                attention_kernel="decomposed",
                head_count_attn=n_heads,
                cache_quantizer=None,
                fake_quant=False,
                scale=sm_scale,
                sliding_window=sliding_window,
                sink=ref_sinks,
            )
            pa_out = pa_out.squeeze(0).permute(1, 0, 2)
            pa_out = pa_out.reshape(seq_len, n_kv_heads, q_mult, head_dim)

        torch.testing.assert_close(
            ref_out,
            pa_out,
            rtol=2e-2,
            atol=2e-2,
            msg="SDPA implementations should match",
        )

    def test_moe_block(self):
        """Test MoE block."""
        shark_moe = self.shark_model.attn_blocks[0].ffn
        ref_moe = self.ref_model.block[0].mlp

        torch.manual_seed(self.seed)
        test_input = torch.tensor(
            [[[0.5, 1.0, -0.5, 0.25], [1.0, 0.5, 0.25, -0.5]]], dtype=torch.bfloat16
        )

        with torch.no_grad():
            shark_moe_out = shark_moe(test_input)

            ref_input = test_input.squeeze(0)
            ref_moe_out = ref_moe(ref_input)
            ref_moe_out = ref_moe_out.unsqueeze(0)

        shark_moe_out_bf16 = shark_moe_out.to(torch.bfloat16)
        torch.testing.assert_close(
            shark_moe_out_bf16, ref_moe_out, rtol=1e-2, atol=1e-2
        )

        self.logger.debug(
            f"MoE - Experts: {self.hp.expert_count}, Used: {self.hp.expert_used_count}"
        )

    def test_e2e_prefill_cross_entropy(self):
        """Compare sharktank vs reference prefill cross-entropy.

        Manual calculation ce and ppl: calculate_cross_entropy_manual(instance, self.sequence, use_prefill=False)
        """

        shark_ce = 3.3252
        shark_ppl = 27.8750

        with torch.no_grad():
            input_ids = torch.tensor([self.test_sequence], dtype=torch.long)
            ref_input = input_ids.squeeze(0)
            ref_logits = self.ref_model(ref_input)

            shift_logits = ref_logits[:-1, :].contiguous()
            shift_labels = ref_input[1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            ref_ce = loss_fct(shift_logits, shift_labels).item()
            ref_ppl = float(torch.exp(torch.tensor(ref_ce)))

        self.logger.info(
            f"Prefill CE  - Sharktank: {shark_ce:.4f}, Reference: {ref_ce:.4f}, Diff: {abs(shark_ce - ref_ce):.4f}"
        )
        self.logger.info(
            f"Prefill PPL - Sharktank: {shark_ppl:.4f}, Reference: {ref_ppl:.4f}, Diff: {abs(shark_ppl - ref_ppl):.4f}"
        )

        torch.testing.assert_close(
            torch.tensor(shark_ce),
            torch.tensor(ref_ce),
            rtol=0.15,
            atol=0.15,
            msg=f"Prefill CE mismatch: shark={shark_ce:.4f} vs ref={ref_ce:.4f}",
        )

        torch.testing.assert_close(
            torch.tensor(shark_ppl),
            torch.tensor(ref_ppl),
            rtol=0.15,
            atol=0.15,
            msg=f"Prefill PPL mismatch: shark={shark_ppl:.4f} vs ref={ref_ppl:.4f}",
        )

    def test_e2e_decode_cross_entropy(self):
        """Compare sharktank vs reference decode cross-entropy.

        Manual calculation ce and ppl: calculate_cross_entropy_manual(instance, self.sequence, use_prefill=False)
        """

        shark_ce = 3.4136
        shark_ppl = 30.1250

        with torch.no_grad():
            total_loss = 0.0
            count = 0

            for i in range(1, len(self.test_sequence)):
                prefix = self.test_sequence[:i]
                target = self.test_sequence[i]

                ref_input = torch.tensor(prefix, dtype=torch.long)
                ref_logits = self.ref_model(ref_input)
                last_logits = ref_logits[-1, :]

                log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
                token_loss = -log_probs[target].item()

                total_loss += token_loss
                count += 1

            ref_ce = total_loss / count
            ref_ppl = float(torch.exp(torch.tensor(ref_ce)))

        self.logger.info(
            f"Decode CE  - Sharktank: {shark_ce:.4f}, Reference: {ref_ce:.4f}, Diff: {abs(shark_ce - ref_ce):.4f}"
        )
        self.logger.info(
            f"Decode PPL - Sharktank: {shark_ppl:.4f}, Reference: {ref_ppl:.4f}, Diff: {abs(shark_ppl - ref_ppl):.4f}"
        )

        torch.testing.assert_close(
            torch.tensor(shark_ce),
            torch.tensor(ref_ce),
            rtol=0.15,
            atol=0.15,
            msg=f"Decode CE mismatch: shark={shark_ce:.4f} vs ref={ref_ce:.4f}",
        )

        torch.testing.assert_close(
            torch.tensor(shark_ppl),
            torch.tensor(ref_ppl),
            rtol=0.15,
            atol=0.15,
            msg=f"Decode PPL mismatch: shark={shark_ppl:.4f} vs ref={ref_ppl:.4f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
