# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from safetensors.torch import save_file
from subprocess import check_output
import sys
import torch


def test_list_safetensors(tmp_path: Path):
    tensor1 = torch.ones([1, 2], dtype=torch.float32)
    tensor2 = torch.ones([3, 4, 5], dtype=torch.int32)

    file_name = tmp_path / "file.safetensors"
    save_file(
        filename=file_name,
        tensors={
            "tensor1": tensor1,
            "tensor2": tensor2,
        },
    )

    output = check_output(
        [
            sys.executable,
            "-m",
            "amdsharktank.tools.list_safetensors",
            "--show-metadata",
            str(file_name),
        ]
    ).decode()
    assert "tensor1: shape=[1, 2], dtype=torch.float32" in output
    assert "tensor2: shape=[3, 4, 5], dtype=torch.int32" in output
