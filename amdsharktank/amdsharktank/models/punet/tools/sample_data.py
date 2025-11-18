# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Various utilities for deriving stable sample data for the model."""

from pathlib import Path

import torch


def get_random_inputs(dtype, device, bs: int = 1):
    torch.random.manual_seed(42)
    max_length = 64
    height = 1024
    width = 1024
    init_batch_dim = 2
    return {
        "sample": torch.rand(bs, 4, height // 8, width // 8, dtype=dtype).to(device),
        "timestep": torch.zeros(1, dtype=torch.int32).to(device),
        "encoder_hidden_states": torch.rand(
            init_batch_dim * bs, max_length, 2048, dtype=dtype
        ).to(device),
        "text_embeds": torch.rand(init_batch_dim * bs, 1280, dtype=dtype).to(device),
        "time_ids": torch.zeros(init_batch_dim * bs, 6, dtype=dtype).to(device),
        "guidance_scale": torch.tensor([7.5], dtype=dtype).to(device),
    }


def load_inputs(st_path: Path, dtype, device, bs: int = 1):
    from safetensors import safe_open

    with safe_open(st_path, framework="pt", device=device) as st:
        random_inputs = get_random_inputs(dtype=dtype, device=device, bs=bs)
        inputs = {}
        for name, random_input in random_inputs.items():
            if name in st.keys():
                print(f"Using specified input for tensor {name}")
                t = st.get_tensor(name)
                inputs[name] = t
            else:
                print(f"Using default/random tensor for tensor {name}")
                inputs[name] = random_input
    return inputs


def save_outputs(st_path: Path, outputs):
    from safetensors.torch import save_file

    tensors = {str(i): t for i, t in enumerate(outputs)}
    save_file(tensors, st_path)
