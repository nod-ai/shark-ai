# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch

from pathlib import Path
from amdsharktank.examples import paged_llm_v1
from amdsharktank.models.llama.testing import make_random_llama_theta
from amdsharktank.models.llama import toy_llama
from amdsharktank.types import Dataset
from amdsharktank.utils.testing import is_mi300x


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Run only on GPU for fast execution."
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
    ],
)
def test_smoke_paged_llm_v1(dtype: torch.dtype, tmp_path: Path):
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
            "--device=cuda:0",
        ]
    )
