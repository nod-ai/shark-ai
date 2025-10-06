# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest
from sharktank.utils._helpers import run_iree_vs_torch_fx
from sharktank.utils._iree_compile_flags_config import LLM_HIP_COMPILE_FLAGS
from sharktank.utils.testing import is_hip_condition
from sharktank.models.wan import WanConfig
from sharktank.models.wan.layers import *
from sharktank.models.wan.testing import (
    make_random_ffnemb_theta,
    make_random_head_theta,
    make_random_mlpproj_theta,
    make_random_time_guidance_projector_theta,
    make_wan_attn_block_random_theta,
)

config = WanConfig()
bs = 1

# @pytest.mark.skipif(f"not ({is_hip_condition})", reason="Test requires HIP device")
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.float16, 1e-4)])
def test_wan_t2v_attn_block_mock_iree_vs_eager(dtype, atol):
    torch.manual_seed(42)
    theta = make_wan_attn_block_random_theta()
    m = WanAttentionBlock(
        theta,
        cross_attn_type="t2v_cross_attn",
        dim=config.dim,
        ffn_dim=config.ffn_dim,
        num_heads=config.num_heads,
        window_size=config.window_size,
        qk_norm=config.qk_norm,
        cross_attn_norm=config.cross_attn_norm,
        eps=config.eps,
        dtype=dtype,
    )
    context_lens = [config.text_len] * bs
    input_args = (
        torch.randn(bs, 21504, config.dim, dtype=dtype),  # x
        torch.randn(bs, 6, config.dim, dtype=dtype),  # e
        torch.tensor([config.text_len], dtype=torch.int),  # seq_lens
        torch.tensor([[21, 32, 32]], dtype=torch.int),  # grid_sizes
        torch.randn(1024, config.dim // config.num_heads // 2, 2, dtype=dtype),  # freqs
        torch.randn(bs, config.text_len, config.dim, dtype=dtype),  # context
        torch.tensor(context_lens, dtype=torch.int),  # context_lens
    )
    run_iree_vs_torch_fx(
        m, input_args=input_args, atol=atol, rtol=0, compile_flags=LLM_HIP_COMPILE_FLAGS
    )
