"""Tests for toy GPT-OSS model generation and inference."""

import torch
import unittest
import logging
from sharktank.models.gpt_oss.toy_gpt_oss import generate, copy_weights_to_reference
from sharktank.utils.llm_utils import (
    LlmInstance,
    TorchInstance,
    llama_config_page_sizes,
)
from sharktank.models.gpt_oss.orig_pytorch_model import ModelConfig, Transformer


class ToyGptOssTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.bfloat16)
        self.seed = 12345

        # Hardcoded for CI performance - regenerate with self.generate_sequence() if weights change
        self.sequence = [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        self.initialized_model()

    def initialized_model(self):

        theta, config = generate(self.seed)

        model = TorchInstance(theta=theta, config=config)
        page_sizes = llama_config_page_sizes(config)
        block_count = 128
        self.shark_instance = LlmInstance(
            model_instance=model,
            page_sizes=page_sizes,
            block_seq_stride=config.block_seq_stride,
            block_count=block_count,
        )

    def generate_sequence(self):
        """Generate test sequence dynamically. Repetitive output expected with random weights."""
        theta, config = generate(self.seed)
        model = TorchInstance(theta=theta, config=config)
        page_sizes = llama_config_page_sizes(config)
        block_count = 128
        instance = LlmInstance(
            model_instance=model,
            page_sizes=page_sizes,
            block_seq_stride=config.block_seq_stride,
            block_count=block_count,
        )
        decoder = instance.make_decoder()
        generated_tokens = decoder.greedy_decode([[0]], steps=14)[0]

        full_sequence = [0] + generated_tokens
        print(f"Generated tokens: {generated_tokens}")
        print(f"Full test sequence: {full_sequence}")
        return full_sequence

    def testDecodeSequence(self):
        """Test deterministic token generation."""

        decoder = self.shark_instance.make_decoder()
        expected = self.sequence[1:]

        decoded = decoder.greedy_decode([[0]], steps=len(expected))[0]
        decoded2 = decoder.greedy_decode([[0]], steps=len(expected))[0]

        self.assertEqual(decoded, decoded2)
        self.assertEqual(decoded, expected)

    def testPrefillPerplexity(self):
        """Test prefill perplexity calculation.
        Manual calculation ce and ppl: calculate_cross_entropy_manual(instance, self.sequence, use_prefill=True)
        """

        decoder = self.shark_instance.make_perplexity_eval()
        result = decoder.prefill_cross_entropy([self.sequence])[0]
        assert result.valid

        shark_ce = 4.6970133781433105
        torch.testing.assert_close(result.score, shark_ce, atol=1e-2, rtol=1e-2)

        result2 = decoder.prefill_cross_entropy([self.sequence])[0]
        self.assertEqual(result.score, result2.score)

    def testDecodePerplexity(self):
        """Test decode perplexity calculation.
        Manual calculation ce and ppl: calculate_cross_entropy_manual(instance, self.sequence, use_prefill=False)
        """

        decoder = self.shark_instance.make_perplexity_eval()
        result = decoder.decode_cross_entropy([self.sequence])[0]
        assert result.valid

        shark_ce = 4.6970133781433105
        torch.testing.assert_close(result.score, shark_ce, atol=1e-2, rtol=1e-2)

        result2 = decoder.decode_cross_entropy([self.sequence])[0]
        self.assertEqual(result.score, result2.score)


class RefSharktankE2ETest(unittest.TestCase):
    """Test reference and sharktank model e2e comparison."""

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        torch.set_default_dtype(torch.bfloat16)
        self.seed = 12345
        torch.manual_seed(self.seed)

        self.initialized_model()
        # Hardcoded for CI performance - regenerate with self.generate_sequence() if weights change
        self.sequence = [0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

    def initialized_model(self):

        # Initialize sharktank model
        self.shark_theta, self.shark_config = generate(self.seed)
        self.hp = self.shark_config.hp
        model = TorchInstance(theta=self.shark_theta, config=self.shark_config)
        page_sizes = llama_config_page_sizes(self.shark_config)
        block_count = 128
        instance = LlmInstance(
            model_instance=model,
            page_sizes=page_sizes,
            block_seq_stride=self.shark_config.block_seq_stride,
            block_count=block_count,
        )
        self.shark_instance = instance

        # Configure reference model
        expert_weight_sample = self.shark_theta(
            "blk", 0, "ffn_gate_exps", "weight"
        ).as_torch()
        actual_intermediate_size = expert_weight_sample.shape[1]

        ref_config = ModelConfig(
            num_hidden_layers=self.hp.block_count,
            num_experts=self.hp.expert_count,
            experts_per_token=self.hp.expert_used_count,
            vocab_size=self.hp.vocab_size,
            hidden_size=self.hp.embedding_length,
            intermediate_size=actual_intermediate_size,
            head_dim=self.hp.attn_head_dim,
            num_attention_heads=self.hp.attention_head_count,
            num_key_value_heads=self.hp.attention_head_count_kv,
            sliding_window=self.hp.sliding_window if self.hp.sliding_window else 128,
            initial_context_length=self.hp.context_length,
            rope_theta=self.hp.rope_freq_base,
            rope_scaling_factor=1.0,
        )

        self.ref_model = Transformer(ref_config, device=torch.device("cpu"))
        self.ref_model.eval()

        copy_weights_to_reference(self.shark_theta, self.ref_model, self.hp)

    def test_ref_sharktank_prefill_cross_entropy(self):
        """Test prefill cross-entropy matches expected values
        Manual calculation ce and ppl: calculate_cross_entropy_manual(instance, self.sequence, use_prefill=True)
        """

        decoder = self.shark_instance.make_perplexity_eval()
        shark_result = decoder.prefill_cross_entropy([self.sequence])[0]
        assert shark_result.valid
        expected_ce = 4.6970133781433105

        with torch.no_grad():
            input_ids = torch.tensor([self.sequence], dtype=torch.long)
            ref_input = input_ids.squeeze(0)
            ref_logits = self.ref_model(ref_input)

            shift_logits = ref_logits[:-1, :].contiguous()
            shift_labels = ref_input[1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            ref_ce = loss_fct(shift_logits, shift_labels).item()
            ref_ppl = float(torch.exp(torch.tensor(ref_ce)))

        torch.testing.assert_close(
            shark_result.score, expected_ce, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(ref_ce, expected_ce, atol=1e-2, rtol=1e-2)

    def test_ref_sharktank_decode_cross_entropy(self):
        """Test decode cross-entropy matches expected values
        Manual calculation ce and ppl: calculate_cross_entropy_manual(instance, self.sequence, use_prefill=False)
        """
        decoder = self.shark_instance.make_perplexity_eval()
        shark_result = decoder.decode_cross_entropy([self.sequence])[0]
        assert shark_result.valid
        expected_ce = 4.6970133781433105

        with torch.no_grad():
            total_loss = 0.0
            count = 0

            for i in range(1, len(self.sequence)):
                prefix = self.sequence[:i]
                target = self.sequence[i]

                ref_input = torch.tensor(prefix, dtype=torch.long)
                ref_logits = self.ref_model(ref_input)

                last_logits = ref_logits[-1, :]
                log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
                token_loss = -log_probs[target].item()

                total_loss += token_loss
                count += 1

            ref_ce = total_loss / count
            ref_ppl = float(torch.exp(torch.tensor(ref_ce)))

        torch.testing.assert_close(
            shark_result.score, expected_ce, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(ref_ce, expected_ce, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
