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

from sharktank.utils import cli
from sharktank.layers import *
from sharktank import ops
from sharktank.types import *
from sharktank.types.pipelining import pipeline_parallelize_theta

from iree.turbine.aot import DeviceAffinity, export


def create_theta(dim: int, shard_count: int, num_layers: int, save_path):
    split_size = dim // shard_count
    weights = []
    for layer in range(num_layers):
        _shard = torch.rand(dim, dim, dtype=torch.float16) / math.sqrt(dim)
        weights.append(
            SplitPrimitiveTensor(
                name=f"blk.{layer}.ffn.weight",
                shard_dim=1,
                ts=_shard.split(split_size, dim=1),
            )
            if shard_count > 1
            else DefaultPrimitiveTensor(name=f"block.{layer}.ffn.weight", data=_shard)
        )

    ones = torch.ones(dim, dim, dtype=torch.float16)
    weights.append(
        SplitPrimitiveTensor(
            name="token_embd.weight",
            shard_dim=1,
            ts=ones.split(split_size, dim=1),
        )
        if shard_count > 1
        else DefaultPrimitiveTensor(name="token_embd.weight", data=ones)
    )
    weights.append(
        SplitPrimitiveTensor(
            name="output.weight",
            shard_dim=1,
            ts=ones.split(split_size, dim=1),
        )
        if shard_count > 1
        else DefaultPrimitiveTensor(name="output.weight", data=ones)
    )
    ones = torch.ones(1, dim, dtype=torch.float16)
    weights.append(
        SplitPrimitiveTensor(
            name="output_norm.weight",
            shard_dim=1,
            ts=ones.split(split_size, dim=1),
        )
        if shard_count > 1
        else DefaultPrimitiveTensor(name="output_norm.weight", data=ones)
    )

    ds = Dataset({}, Theta(weights))
    ds.save(save_path)


class PPFFN(ThetaLayer):
    block_to_pipeline: tuple[int, ...]
    pipeline_to_devices: tuple[list[int], ...]

    def __init__(
        self,
        theta,
        block_to_pipeline: tuple[int, ...],
        pipeline_to_devices: tuple[list[int], ...],
    ):
        super().__init__(theta)
        self.block_to_pipeline = block_to_pipeline
        self.pipeline_to_devices = pipeline_to_devices

    def _inter_layer_callback(self, x: ShardedTensor, curr_block: int):
        if self.block_to_pipeline is None:
            return x

        if curr_block >= len(self.block_to_pipeline) - 1:
            return x

        curr_pipeline = self.block_to_pipeline[curr_block]
        next_pipeline = self.block_to_pipeline[curr_block + 1]

        curr_devices = self.pipeline_to_devices[curr_pipeline]
        next_devices = self.pipeline_to_devices[next_pipeline]

        if all(d_curr == d_next for d_curr, d_next in zip(curr_devices, next_devices)):
            return x

        shards = ShardedTensor.move_shards_to_new_devices(
            x.shards, old_devices=curr_devices, new_devices=next_devices
        )
        return x.clone(ts=shards, devices=next_devices)

    def forward(self, x: torch.Tensor):
        num_blocks = len(self.block_to_pipeline)
        shard_count = self.theta.tensor("blk.0.ffn.weight").shard_count

        x = ReplicatedTensor(
            ts=x, shard_count=shard_count, devices=self.pipeline_to_devices[0]
        )
        for block in range(num_blocks):
            weight = self.theta.tensor(f"blk.{block}.ffn.weight")
            x = ops.replicate(ops.linear(x, weight), shard_count)
            x = self._inter_layer_callback(x, block)

        return ops.unshard(x)


def main(raw_args=None):
    parser = cli.create_parser()
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

    bs = 16
    sl = 128
    primary_dim = 128 * 2**5
    shard_count = 2
    num_layers = 40
    create_theta(primary_dim, shard_count, num_layers, save_path=args.output_irpa_file)

    pp_count = 4
    ds = Dataset.load(args.output_irpa_file)
    block_to_pipeline, pipeline_to_devices = pipeline_parallelize_theta(
        ds.root_theta, pp_count
    )

    mdl = PPFFN(ds.root_theta, block_to_pipeline, pipeline_to_devices)

    example_arg = torch.empty(bs, sl, primary_dim, dtype=torch.float16)
    ep = torch.export.export(mdl, (example_arg,), strict=False)
    cm = export(ep, arg_device={0: DeviceAffinity(str(pipeline_to_devices[0][0]))})

    if args.output_file == "-":
        print(cm.mlir_module)
    else:
        with open(args.output_file, "wt") as f:
            f.write(str(cm.mlir_module))


if __name__ == "__main__":
    main()
