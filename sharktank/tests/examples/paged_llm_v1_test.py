# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from pathlib import Path
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.models.llama import toy_llama
from sharktank.types import Dataset
from sharktank.examples import paged_llm_v1


@pytest.mark.parametrize(
    "dtype,tensor_parallelism_size",
    [
        (torch.float16, 1),
        (torch.float16, 2),
        (torch.float32, 1),
        (torch.float32, 2),
    ],
)
def test_smoke_paged_llm_v1(
    dtype: torch.dtype, tensor_parallelism_size: int, tmp_path: Path
):
    theta, config = toy_llama.generate(seed=0, dtype_rest=dtype, dtype_norm=dtype)
    dataset = Dataset(root_theta=theta, properties=config.to_properties())
    dataset_path = tmp_path / "model.irpa"
    dataset.save(dataset_path)
    paged_llm_v1.main(
        [
            f"--irpa-file={dataset_path}",
            "--tokenizer-type=fake",
            "--bs=4",
            "--prompt-seq-len=2",
            "--max-decode-steps=1",
            f"--tensor-parallelism-size={tensor_parallelism_size}",
        ]
    )
