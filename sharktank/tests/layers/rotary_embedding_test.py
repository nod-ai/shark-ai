import torch

from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers import LlamaConfig

import pytest


class HFRotaryEmbedding(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self._rotary = LlamaRotaryEmbedding(config=config)

    def forward(self, *, xt, positions):
        cos, sin = self._rotary(xt, positions)
        xt = xt.transpose(1, 2)
        return apply_rotary_pos_emb(xt, xt, cos, sin)[0].transpose(1, 2)


bs = 2
length = 5
heads = 3
dims = 128
rope_scaling = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}

hf_config = LlamaConfig(
    max_position_embeddings=131072,
    rope_theta=500000,
)

torch.manual_seed(1234567)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="kernel implementation for bfloat16 is inacurate due to unidentified bug"
            ),
        ),
    ],
)
def test_rope(dtype: torch.dtype):

    hf_rotary = HFRotaryEmbedding(config=hf_config)

    st_torch_rotary = RotaryEmbeddingLayer(
        rope_dimension_count=dims,
        max_seqlen=2048,
        rope_freq_base=500000,
        use_hf=True,
        dtype=dtype,
    )

    st_kernel_rotary = RotaryEmbeddingLayer(
        rope_dimension_count=dims,
        max_seqlen=2048,
        rope_freq_base=500000,
        use_hf=False,
        dtype=dtype,
    )

    def test_prefill(st_rotary: RotaryEmbeddingLayer):
        example = torch.rand(bs, length, heads, dims, dtype=dtype)
        positions = torch.arange(0, length)[None, :].repeat(bs, 1)
        hf_results = hf_rotary(xt=example, positions=positions)
        st_results = st_rotary.forward(xt=example, start_index=0)
        torch.testing.assert_close(st_results, hf_results)

    def test_decode(st_rotary: RotaryEmbeddingLayer):
        decode_example = torch.rand(bs, 1, heads, dims, dtype=dtype)

        hf_results = hf_rotary.forward(
            xt=decode_example, positions=torch.arange(0, bs).unsqueeze(1)
        )

        st_mask = st_rotary.compute_batch_mask(
            start_positions=torch.arange(0, bs), batch_seq_len=1
        )
        st_results = st_rotary.apply_batched_mask_unsharded(
            xt=decode_example, mask=st_mask
        )

        torch.testing.assert_close(st_results, hf_results)

    test_prefill(st_torch_rotary)
    test_prefill(st_kernel_rotary)
    test_decode(st_torch_rotary)
    test_decode(st_kernel_rotary)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(
                reason="kernel implementation for bfloat16 is inacurate due to unidentified bug"
            ),
        ),
    ],
)
def test_rope_llama3_scaling(dtype: torch.dtype):
    hf_config = LlamaConfig(
        rope_scaling=rope_scaling,
        max_position_embeddings=131072,
        rope_theta=500000,
    )

    hf_rotary = HFRotaryEmbedding(config=hf_config)

    st_torch_rotary = RotaryEmbeddingLayer(
        rope_dimension_count=dims,
        max_seqlen=2048,
        rope_freq_base=500000,
        use_hf=True,
        rope_scaling_type="llama3",
        dtype=dtype,
    )

    st_kernel_rotary = RotaryEmbeddingLayer(
        rope_dimension_count=dims,
        max_seqlen=2048,
        rope_freq_base=500000,
        use_hf=False,
        rope_scaling_type="llama3",
        dtype=dtype,
    )

    def test_prefill(st_rotary: RotaryEmbeddingLayer):
        example = torch.rand(bs, length, heads, dims, dtype=dtype)
        positions = torch.arange(0, length)[None, :].repeat(bs, 1)
        hf_results = hf_rotary(xt=example, positions=positions)
        st_results = st_rotary.forward(xt=example, start_index=0)
        torch.testing.assert_close(st_results, hf_results)

    def test_decode(st_rotary: RotaryEmbeddingLayer):
        decode_example = torch.rand(bs, 1, heads, dims, dtype=dtype)

        hf_results = hf_rotary.forward(
            xt=decode_example, positions=torch.arange(0, bs).unsqueeze(1)
        )

        st_mask = st_rotary.compute_batch_mask(
            start_positions=torch.arange(0, bs), batch_seq_len=1
        )
        st_results = st_rotary.apply_batched_mask_unsharded(
            xt=decode_example, mask=st_mask
        )

        torch.testing.assert_close(st_results, hf_results)

    test_prefill(st_torch_rotary)
    test_prefill(st_kernel_rotary)
    test_decode(st_torch_rotary)
    test_decode(st_kernel_rotary)
