# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from safetensors import safe_open
import torch
import unittest
import pytest


@pytest.mark.skip(reason="need to generate values to compare against")
class QuarkParityTest(unittest.TestCase):
    def test_compare_against_quark(self):
        def both(key, index=None):
            o = ours[key]
            t = theirs[key]
            if index is None:
                return o, t
            else:
                return o[index], t[index]

        mapping = dict()
        for i in range(32):
            hf = f"model.layers.{i}"
            gg = f"attn_blocks.{i}"
            base_pairs = [
                [f"{hf}.input_layernorm", f"{gg}.attn.attn_norm"],
                [f"{hf}.self_attn.k_proj", f"{gg}.attn.attn_k"],
                [f"{hf}.self_attn.q_proj", f"{gg}.attn.attn_q"],
                [f"{hf}.self_attn.v_proj", f"{gg}.attn.attn_v"],
                [f"{hf}.self_attn.o_proj", f"{gg}.attn.attn_output"],
                [f"{hf}.post_attention_layernorm", f"{gg}.ffn_norm"],
                [f"{hf}.mlp.down_proj", f"{gg}.ffn.ffn_down"],
                [f"{hf}.mlp.gate_proj", f"{gg}.ffn.ffn_gate"],
                [f"{hf}.mlp.up_proj", f"{gg}.ffn.ffn_up"],
            ]
            for a, b in base_pairs:
                mapping[a] = b
                mapping[a + "_input_0"] = b + "_input_0"

        ours = dict()
        with safe_open("../ours_newest_prefill.safetensors", "pytorch") as st:
            for key in st.keys():
                ours[key] = st.get_tensor(key)

        theirs = dict()
        with safe_open("../theirs2.safetensors", "pytorch") as st:
            for key in st.keys():
                if key in mapping:
                    theirs[mapping[key]] = st.get_tensor(key)

        test_layers = [v for k, v in mapping.items()]
        for lyr in test_layers:
            name = lyr
            if name in ours.keys() and name != "freqs":
                o, t = both(name)
                torch.testing.assert_close(o, t, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
