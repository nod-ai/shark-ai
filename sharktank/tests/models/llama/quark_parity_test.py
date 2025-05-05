# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os

from safetensors import safe_open
import torch
import unittest
import pytest
from pathlib import Path
import subprocess
from sharktank.utils.testing import TempDirTestBase

with_quark_data = pytest.mark.skipif("not config.getoption('with_quark_data')")


class QuarkParityTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        self.path_prefix = Path("/shark-cache/quark_test")

    @pytest.mark.xfail(
        reason="Known accuracy validation issues with quark parity. See https://github.com/nod-ai/shark-ai/issues/1051"
    )
    @with_quark_data
    def test_compare_against_quark(self):
        sharktank_dir = str(
            Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent.parent
        )
        our_path = self._temp_dir / "ours_prefill.safetensors"
        if os.path.exists(our_path):
            os.remove(our_path)

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

        command = [
            "python",
            "-m",
            "sharktank.examples.paged_llm_v1",
            "The capitol of Texas is",
            f"--irpa-file={self.path_prefix}/fp8_bf16_weight.irpa",
            f"--tokenizer-config-json=/shark-dev/data/llama3.1/8b/tokenizer.json",
            "--fake-quant",
            "--attention-kernel=torch",
            "--activation-dtype=bfloat16",
            f"--save_intermediates_path={self._temp_dir / 'ours'}",
            "--use-hf",
            "--attention-dtype=bfloat16",
            "--kv-cache-dtype=float8_e4m3fnuz",
            "--skip-decode",
            "--block-seq-stride=16",
        ]
        command = subprocess.list2cmdline(command)
        subprocess.check_call(command, shell=True, cwd=sharktank_dir)

        ours = dict()
        with safe_open(our_path, "pytorch") as st:
            for key in st.keys():
                ours[key] = st.get_tensor(key)

        golden = dict()
        golden_path = self.path_prefix / "golden.safetensors"
        with safe_open(golden_path, "pytorch") as st:
            for key in st.keys():
                if key in mapping:
                    golden[mapping[key]] = st.get_tensor(key)

        test_layers = [v for k, v in mapping.items()]

        def both(key, index=None):
            o = ours[key]
            t = golden[key]
            if index is None:
                return o, t
            else:
                return o[index], t[index]

        for lyr in test_layers:
            name = lyr
            if name in ours.keys() and name != "freqs":
                o, t = both(name)
                torch.testing.assert_close(o, t, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
