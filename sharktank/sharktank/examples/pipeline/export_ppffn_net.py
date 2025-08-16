# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Example program to export a sharded and pipeline parallized set FFN networks.
This is used for developing and testing various tooling flows with a scaled down example.

Generate MLIR and a random inited IRPA file with:

    python -m sharktank.examples.sharding.export_pffn_net \
        --output-irpa-file=/tmp/ffn.irpa /tmp/ffn.mlir
"""

import os
import math

import torch

from typing import Any
from sharktank.utils import cli
from sharktank.layers import *
from sharktank import ops
from sharktank.types import *
from sharktank.types.pipelining import (
    default_distribute_blocks_for_pipeline_parallelism,
    parallelize_in_place,
)

from iree.turbine.aot import DeviceAffinity, export


def create_theta(
    dim: int, tensor_parallelism_size: int, num_layers: int, save_path
) -> Theta:
    split_size = dim // tensor_parallelism_size
    weights = []
    for layer in range(num_layers):
        _shard = torch.rand(dim, dim, dtype=torch.float16) / math.sqrt(dim)
        weights.append(
            SplitPrimitiveTensor(
                name=f"blk.{layer}.ffn.weight",
                shard_dim=1,
                ts=_shard.split(split_size, dim=1),
            )
            if tensor_parallelism_size > 1
            else DefaultPrimitiveTensor(name=f"blk.{layer}.ffn.weight", data=_shard)
        )

    return Theta(weights)


def pipeline_parallelize_theta(
    theta: Theta, pipeline_parallelism_size: int
) -> tuple[list[int] | None, list[list[int]] | None]:
    """
    Pipeline parallelize theta for LLM.
    Both DeepSeek and Llama.
    """
    if pipeline_parallelism_size == 1:
        return None, None

    sample_weight = theta("blk.0.ffn.weight")
    tensor_parallelism_size = (
        sample_weight.shard_count if isinstance(sample_weight, ShardedTensor) else 1
    )

    block_indices = theta.tensor("blk").keys()
    block_count = len(block_indices)

    (
        block_to_pipeline_stage,
        pipeline_stage_to_devices,
    ) = default_distribute_blocks_for_pipeline_parallelism(
        block_count=block_count,
        pipeline_parallelism_size=pipeline_parallelism_size,
        tensor_parallelism_size=tensor_parallelism_size,
    )

    assert (
        bi == i for i, bi in enumerate(block_indices)
    ), "Blocks assumed to be numbered contiguously from [0, N-1]"
    for blk_idx in block_indices:
        blk_idx = int(blk_idx)
        stage = block_to_pipeline_stage[blk_idx]
        devices = pipeline_stage_to_devices[stage]

        block_data = theta.tensor("blk", blk_idx)
        for t_name in block_data.keys():
            parallelize_in_place(block_data[t_name], devices)

    return block_to_pipeline_stage, pipeline_stage_to_devices


class PPFFN(ThetaLayer):
    block_to_pipeline_stage: tuple[int, ...]
    pipeline_stage_to_devices: tuple[list[int], ...]

    def __init__(
        self,
        theta,
        block_to_pipeline_stage: tuple[int, ...],
        pipeline_stage_to_devices: tuple[list[int], ...],
    ):
        super().__init__(theta)
        self.block_to_pipeline_stage = block_to_pipeline_stage
        self.pipeline_stage_to_devices = pipeline_stage_to_devices
        self.blocks = torch.nn.ModuleList(
            LinearLayer(theta(f"blk.{block_idx}.ffn"))
            for block_idx in range(len(block_to_pipeline_stage))
        )

    def _prepare_pipeline_parallel_block_args(
        self, x: torch.Tensor | ReplicatedTensor, block_index: int
    ) -> dict[str, Any]:
        """
        Args:
        -----
        x: pipeline input or previous layer output.
        """

        res = {}
        if block_index == 0:
            res["x"] = ReplicatedTensor(
                ts=x,
                shard_count=len(self.pipeline_stage_to_devices[0]),
                devices=self.pipeline_stage_to_devices[0],
            )
        else:
            stage = self.block_to_pipeline_stage[block_index]
            prev_stage = self.block_to_pipeline_stage[block_index - 1]

            stage_devices = self.pipeline_stage_to_devices[stage]
            prev_stage_devices = self.pipeline_stage_to_devices[prev_stage]

            x = ops.replicate(x, count=x.shard_count)

            if all(
                prev_stage_device == stage_device
                for prev_stage_device, stage_device in zip(
                    prev_stage_devices, stage_devices
                )
            ):
                res["x"] = x
            else:
                shards = ShardedTensor.move_shards_to_new_devices(
                    x.shards, old_devices=prev_stage_devices, new_devices=stage_devices
                )
                res["x"] = x.clone(ts=shards, devices=stage_devices)

        return res

    def forward(self, x: torch.Tensor):
        for block_index, block in enumerate(self.blocks):
            block_kwargs = self._prepare_pipeline_parallel_block_args(x, block_index)
            x = block(**block_kwargs)

        return ops.unshard(x)


def main(raw_args=None):
    parser = cli.create_parser()
    parser.add_argument(
        "--tensor-parallelism-size",
        type=int,
        default=1,
        help="Number of tensor shards for tensor parallelism.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default="-",
        help="Output file to save MLIR to",
    )
    cli.add_output_dataset_options(parser)
    args = cli.parse(parser, args=raw_args)

    if args.output_irpa_file and args.output_irpa_file != "-":
        irpa_dir = os.path.dirname(args.output_irpa_file)
        if irpa_dir and not os.path.exists(irpa_dir):
            raise ValueError(
                f"Parent directory for output IRPA file does not exist: {irpa_dir}"
            )
    if args.output_file and args.output_file != "-":
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(
                f"Parent directory for output file does not exist: {output_dir}"
            )

    bs = 11
    input_dim = 13
    ffn_hidden_dim = 5 * args.tensor_parallelism_size
    pp_count = 3
    num_layers = pp_count * 2
    theta = create_theta(
        ffn_hidden_dim,
        args.tensor_parallelism_size,
        num_layers,
        save_path=args.output_irpa_file,
    )

    block_to_pipeline_stage, pipeline_stage_to_devices = pipeline_parallelize_theta(
        theta, pp_count
    )
    Dataset({}, theta).save(args.output_irpa_file)
    ds = Dataset.load(args.output_irpa_file)

    mdl = PPFFN(ds.root_theta, block_to_pipeline_stage, pipeline_stage_to_devices)

    example_arg = torch.empty(bs, input_dim, ffn_hidden_dim, dtype=torch.float16)
    ep = torch.export.export(mdl, (example_arg,), strict=False)
    cm = export(ep, arg_device={0: DeviceAffinity(pipeline_stage_to_devices[0][0])})

    if args.output_file == "-":
        print(cm.mlir_module)
    else:
        with open(args.output_file, "wt") as f:
            f.write(str(cm.mlir_module))


if __name__ == "__main__":
    main()
