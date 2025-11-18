# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import torch

from amdsharktank.types import (
    Dataset,
    Theta,
    DefaultPrimitiveTensor,
    ReplicatedTensor,
    SplitPrimitiveTensor,
    unbox_tensor,
)


def test_import_export_json(tmp_path: Path):
    src_dataset = Dataset(
        properties={
            "test_property": 1,
        },
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
                "c": SplitPrimitiveTensor(
                    shard_dim=0,
                    ts=[torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])],
                    name="c",
                ),
            }
        ),
    )

    json_path = tmp_path / "dataset.json"
    src_dataset.save(json_path, file_type="json")
    assert json_path.exists()
    new_dataset = Dataset.load(json_path, file_type="json")

    assert "test_property" in new_dataset.properties
    assert new_dataset.properties["test_property"] == 1

    inference_tensors = new_dataset.root_theta.flatten()

    assert "a" in inference_tensors
    assert isinstance(inference_tensors["a"], DefaultPrimitiveTensor)
    assert "b" in inference_tensors
    assert isinstance(inference_tensors["b"], ReplicatedTensor)
    assert "c" in inference_tensors
    assert isinstance(inference_tensors["c"], SplitPrimitiveTensor)
