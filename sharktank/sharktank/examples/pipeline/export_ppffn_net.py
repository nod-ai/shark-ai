# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Example program to export a sharded FFN network like what is found in
a typical transformer layer. This is used for developing and testing various
tooling flows with a scaled down example.

Generate MLIR and a random inited IRPA file with:

    python -m sharktank.examples.sharding.export_ffn_net \
        --output-irpa-file=/tmp/ffn.irpa /tmp/ffn.mlir
"""

import math

import torch

from ...layers import *
from ... import ops
from ...types import *

from iree.turbine.aot import DeviceAffinity, DeviceTensorTrait, export


def create_theta(dim: int, shard_count: int, num_layers: int, save_path):
    split_size = dim // shard_count
    weights = []
    for layer in range(num_layers):
        _weight = torch.rand(dim, dim, dtype=torch.float16) / math.sqrt(dim)
        weights.append(
            SplitPrimitiveTensor(
                name=f"w.{layer}", shard_dim=1, ts=_weight.split(split_size, dim=1)
            )
        )
    ds = Dataset({}, Theta(weights))
    ds.save(save_path)


def pipeline_parallelize_theta(theta: Theta, pp_count: int) -> Theta:
    num_layers = len(theta.tensor("w"))
    shard_count = theta.tensor("w", "0").shard_count
    for layer in list(theta.tensor("w").keys()):
        weight: ShardedTensor = theta.tensor("w", layer)
        pp_group = int(int(layer) * pp_count / num_layers)
        zero_4_group = shard_count * pp_group
        devices = tuple(i + zero_4_group for i in range(shard_count))

        shards = weight.shards
        for i, shard in enumerate(shards):
            DeviceTensorTrait(devices[i]).set(shard._data)
        theta.tensor("w")[layer] = weight.clone(ts=shards, devices=devices, pinned=True)
    return theta


class PPFFN(ThetaLayer):
    def forward(self, x: torch.Tensor):
        num_layers = len(self.theta.tensor("w"))
        shard_count = self.theta.tensor("w", "0").shard_count

        x = ReplicatedTensor(ts=x, shard_count=shard_count)
        for layer in range(num_layers):
            weight: SplitPrimitiveTensor = self.theta.tensor("w", str(layer))
            x: ReplicatedTensor = ops.all_reduce(ops.linear(x, weight))

        return x


def main(raw_args=None):
    from ...utils import cli

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

    bs = 16
    sl = 128
    primary_dim = 128 * 2**5
    shard_count = 2
    num_layers = 40
    create_theta(primary_dim, shard_count, num_layers, save_path=args.output_irpa_file)

    pp_count = 4
    ds = Dataset.load(args.output_irpa_file)
    root_theta = pipeline_parallelize_theta(ds.root_theta, pp_count)

    mdl = PPFFN(root_theta)

    example_arg = torch.empty(bs, sl, primary_dim, dtype=torch.float16)
    ep = torch.export.export(mdl, (example_arg,))  # , strict=False)
    cm = export(ep, arg_device={0: DeviceAffinity(0)})

    if args.output_file == "-":
        print(cm.mlir_module)
    else:
        with open(args.output_file, "wt") as f:
            f.write(str(cm.mlir_module))


if __name__ == "__main__":
    main()
