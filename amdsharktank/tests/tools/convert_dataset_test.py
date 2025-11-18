# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import torch

from amdsharktank.tools import convert_dataset
from amdsharktank.types import (
    Dataset,
    Theta,
    DefaultPrimitiveTensor,
    ReplicatedTensor,
    unbox_tensor,
)


def test_convert_dataset_dtypes(tmp_path: Path):
    src_dataset = Dataset(
        properties={},
        root_theta=Theta(
            {
                "a": DefaultPrimitiveTensor(
                    data=torch.tensor([1.1, 2.2, 3.3], dtype=torch.float16), name="a"
                ),
                "b": ReplicatedTensor(
                    ts=torch.tensor(
                        [1, 2, 3],
                        dtype=torch.int32,
                    ),
                    name="b",
                    shard_count=2,
                ),
                "c": DefaultPrimitiveTensor(
                    data=torch.tensor([4, 5, 6], dtype=torch.int16), name="c"
                ),
            }
        ),
    )

    src_dataset_path = tmp_path / "src_dataset.irpa"
    src_dataset.save(src_dataset_path)

    tgt_dataset_path = tmp_path / "tgt_dataset.irpa"

    convert_dataset.main(
        [
            f"--irpa-file={src_dataset_path}",
            f"--output-irpa-file={tgt_dataset_path}",
            "--dtype=float16->float32",
            "--dtype=int32->int64",
        ]
    )

    tgt_dataset = Dataset.load(tgt_dataset_path)

    assert tgt_dataset.root_theta("a").name == src_dataset.root_theta("a").name
    assert tgt_dataset.root_theta("a").dtype == torch.float32
    torch.testing.assert_close(
        unbox_tensor(tgt_dataset.root_theta("a")),
        unbox_tensor(src_dataset.root_theta("a")),
        check_dtype=False,
    )

    assert tgt_dataset.root_theta("b").name == src_dataset.root_theta("b").name
    assert tgt_dataset.root_theta("b").dtype == torch.int64
    assert isinstance(tgt_dataset.root_theta("b"), ReplicatedTensor)
    assert tgt_dataset.root_theta("b").shard_count == 2
    torch.testing.assert_close(
        unbox_tensor(tgt_dataset.root_theta("b")),
        unbox_tensor(src_dataset.root_theta("b")),
        check_dtype=False,
    )

    assert tgt_dataset.root_theta("c").name == src_dataset.root_theta("c").name
    assert tgt_dataset.root_theta("c").dtype == torch.int16
    torch.testing.assert_close(
        unbox_tensor(tgt_dataset.root_theta("c")),
        unbox_tensor(src_dataset.root_theta("c")),
    )
