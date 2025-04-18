# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch
from iree.turbine.aot import *
from sharktank.types import Theta
from sharktank.layers.testing import make_random_moe_block_theta
from sharktank.utils.testing import make_rand_torch
from sharktank.layers.mixture_of_experts_block import MoeBlock


class MoeBlockTest(unittest.TestCase):
    def test(self):
        dtype = torch.float32
        batch_size = 3
        seq_len = 5
        in_dim = 7

        theta = make_random_moe_block_theta(
            in_dim=in_dim,
            expert_hidden_dim=13,
            num_experts=17,
            with_ffn_norm=True,
            num_shared_experts=19,
            shared_expert_hidden_dim=23,
            with_layer_output_norm=True,
            dtype=dtype,
        )
        theta.rename_tensors_to_paths()
        model = MoeBlock(
            theta=theta,
            expert_used_count=2,
            rms_epsilon=1e-5,
        )
        fxb = FxProgramsBuilder(model)
        input = make_rand_torch((batch_size, seq_len, in_dim))

        @fxb.export_program(name="moe_block", args=(input,), strict=False)
        def _(model, input: torch.Tensor) -> torch.Tensor:
            return model(input)


if __name__ == "__main__":
    unittest.main()
