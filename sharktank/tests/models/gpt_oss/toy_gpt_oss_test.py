from calendar import c
import pytest
import torch
import unittest

from sharktank.models.gpt_oss.toy_gpt_oss import generate, generate_analytical
from sharktank.utils.llm_utils import (
    LlmInstance,
    TorchInstance,
    llama_config_page_sizes,
)


def calculate_cross_entropy_manual(
    model_instance, sequence: list[int], use_prefill: bool = True
) -> float:

    evaluator = model_instance.make_perplexity_eval()
    if use_prefill:
        res = evaluator.prefill_cross_entropy([sequence])[0]
    else:
        res = evaluator.decode_cross_entropy([sequence])[0]

    assert res.valid
    ce = res.score
    ppl = float(torch.exp(torch.tensor(ce)))

    print("cross_entropy_nats:", ce)
    print("perplexity:", ppl)
    return ce, ppl


class ToyGptOssTest(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.bfloat16)

        self.seed = 12345
        # if we want to generate the sequence dynamically, we can use the following code:
        # self.sequence = self.generate_sequence()
        self.sequence = [0, 46, 52, 121, 73, 76, 81, 104, 127, 34, 121, 1, 106, 22, 103]

    def generate_sequence(self):
        """Generate the test sequence dynamically: [0] + model_generated_tokens
        sequence must be within the vocab range [0, 127]
        """

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
        generated_tokens = decoder.greedy_decode([[0]], steps=15)[0]

        # Return full sequence: [start_token] + [generated_tokens]
        full_sequence = [0] + generated_tokens
        print(f"Generated tokens: {generated_tokens}")
        print(f"Full test sequence: {full_sequence}")
        return full_sequence

    def testDtypeConsistency(self):
        """Test that model weights maintain expected dtypes."""
        torch.set_default_dtype(torch.bfloat16)
        theta, config = generate(self.seed)

        # Test that all weights are in bfloat16 as expected
        token_emb = theta("token_embd.weight").as_torch()
        self.assertEqual(
            token_emb.dtype, torch.bfloat16, "Token embedding should be bfloat16"
        )

        # Test attention weights
        block_0 = theta("blk", 0)
        qkv_weight = block_0("attn", "wqkv", "weight").as_torch()
        self.assertEqual(
            qkv_weight.dtype, torch.bfloat16, "QKV weights should be bfloat16"
        )

        # Test MoE weights
        moe_gate = block_0("ffn_gate_inp", "weight").as_torch()
        self.assertEqual(
            moe_gate.dtype, torch.bfloat16, "MoE gate weights should be bfloat16"
        )

        expert_gate = block_0("ffn_gate_exps", "weight").as_torch()
        self.assertEqual(
            expert_gate.dtype, torch.bfloat16, "Expert gate weights should be bfloat16"
        )

        # Test config dtypes
        self.assertEqual(
            config.activation_dtype,
            torch.bfloat16,
            "Config activation_dtype should be bfloat16",
        )
        self.assertEqual(
            config.attention_dtype,
            torch.bfloat16,
            "Config attention_dtype should be bfloat16",
        )

    def testDecodeSequence(self):
        """Test deterministic token generation (e2e)."""
        torch.set_default_dtype(torch.bfloat16)
        theta, config = generate(self.seed)

        model = TorchInstance(theta=theta, config=config)
        page_sizes = llama_config_page_sizes(config)
        block_count = 128

        instance = LlmInstance(
            model_instance=model,
            block_count=block_count,
            page_sizes=page_sizes,
            block_seq_stride=config.block_seq_stride,
        )

        decoder = instance.make_decoder()

        expected = self.sequence[1:]

        decoded = decoder.greedy_decode([[0]], steps=len(expected))[0]
        decoded2 = decoder.greedy_decode([[0]], steps=len(expected))[0]

        self.assertEqual(decoded, decoded2, "Greedy decode should be deterministic")
        self.assertEqual(decoded, expected, "Greedy decode differs from golden output")

    def testPrefillPerplexity(self):
        """Test prefill perplexity calculation (e2e).
        If we want to calculate the cross entropy manually, we can use the following code:
        ce,ppl = calculate_cross_entropy_manual(instance, self.sequence, use_prefill=True)
        """
        torch.set_default_dtype(torch.bfloat16)
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

        decoder = instance.make_perplexity_eval()
        result = decoder.prefill_cross_entropy([self.sequence])[0]
        assert result.valid, "Prefill perplexity should be valid"

        expected_ce = 0.26273706555366516
        torch.testing.assert_close(
            torch.tensor(result.score, dtype=torch.float32),
            torch.tensor(expected_ce, dtype=torch.float32),
            atol=1e-2,
            rtol=1e-2,
            msg=f"Expected CE {expected_ce}, got {result.score}",
        )

        result2 = decoder.prefill_cross_entropy([self.sequence])[0]
        self.assertEqual(
            result.score, result2.score, "Perplexity should be deterministic"
        )

    def testDecodePerplexity(self):
        """Test decode perplexity calculation (e2e).
        If we want to calculate the cross entropy manually, we can use the following code:
        ce,ppl = calculate_cross_entropy_manual(instance, self.sequence, use_prefill=False)
        """
        torch.set_default_dtype(torch.bfloat16)
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

        decoder = instance.make_perplexity_eval()

        expected_ce = 0.1829063594341278
        result = decoder.decode_cross_entropy([self.sequence])[0]
        self.assertIsInstance(result.score, float)
        result2 = decoder.decode_cross_entropy([self.sequence])[0]
        self.assertEqual(
            result.score, result2.score, "Perplexity should be deterministic"
        )

        torch.testing.assert_close(
            torch.tensor(result.score, dtype=torch.float32),
            torch.tensor(expected_ce, dtype=torch.float32),
            atol=1e-2,
            rtol=1e-2,
            msg=f"Expected CE {expected_ce}, got {result.score}",
        )


if __name__ == "__main__":
    unittest.main()
