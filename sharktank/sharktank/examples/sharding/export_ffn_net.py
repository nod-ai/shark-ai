# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Example program to export a sharded FFN network like what is found in
a typical transformer layer. This is used for developing and testing various
tooling flows with a scaled down example.

Generate MLIR and a random inited IRPA file with:

    python -m amdsharktank.examples.sharding.export_ffn_net \
        --output-irpa-file=/tmp/ffn.irpa /tmp/ffn.mlir
"""
import os

import torch

from amdsharktank.layers import *
from amdsharktank import ops
from amdsharktank.types import *
from amdsharktank.utils import cli


def create_theta(
    hidden_dim: int = 128, primary_dim: int = 64, shard_count: int = 8
) -> Theta:
    split_size = hidden_dim // shard_count
    # Columnwise (transposed) sharding of gate and up weight.
    gate_weight = torch.rand(hidden_dim, primary_dim, dtype=torch.float16)
    up_weight = torch.rand(hidden_dim, primary_dim, dtype=torch.float16)
    sharded_gate_weight = SplitPrimitiveTensor(
        name="ffn_gate.weight", shard_dim=0, ts=gate_weight.split(split_size, dim=0)
    )
    sharded_up_weight = SplitPrimitiveTensor(
        name="ffn_up.weight", shard_dim=0, ts=up_weight.split(split_size, dim=0)
    )

    # Rowwise (transposed) sharding of down weight.
    down_weight = torch.rand(primary_dim, hidden_dim, dtype=torch.float16)
    sharded_down_weight = SplitPrimitiveTensor(
        name="ffn_down.weight", shard_dim=1, ts=down_weight.split(split_size, dim=1)
    )

    return Theta([sharded_gate_weight, sharded_up_weight, sharded_down_weight])


class ShardedFFN(ThetaLayer):
    def forward(self, x: torch.Tensor):
        ffn_gate_weight = self.theta.tensor("ffn_gate", "weight")
        ffn_up_weight = self.theta.tensor("ffn_up", "weight")
        ffn_down_weight = self.theta.tensor("ffn_down", "weight")
        x = ops.replicate(x, count=ffn_gate_weight.shard_count)
        ffn_gate = ops.elementwise(
            torch.nn.functional.silu, ops.linear(x, ffn_gate_weight)
        )
        ffn_up = ops.linear(x, ffn_up_weight)
        ffn_down = ops.linear(
            ops.elementwise(torch.mul, ffn_gate, ffn_up), ffn_down_weight
        )
        summed = ops.sharded_sum(ffn_down)
        return summed


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

    bs = 4
    sl = 32
    hidden_dim = 128
    primary_dim = 64
    shard_count = 8
    theta = create_theta(hidden_dim, primary_dim, shard_count)

    # Roundtrip the dataset, which anchors the tensors as parameters to be loaded
    # vs constants to be frozen (TODO: This is a bit wonky).
    ds = Dataset({}, theta)
    ds.save(args.output_irpa_file)
    ds = Dataset.load(args.output_irpa_file)

    mdl = ShardedFFN(ds.root_theta)
    from iree.turbine import aot

    example_arg = torch.empty(bs, sl, primary_dim, dtype=torch.float16)
    ep = torch.export.export(mdl, (example_arg,))
    cm = aot.export(ep)

    if args.output_file == "-":
        print(cm.mlir_module)
    else:
        with open(args.output_file, "wt") as f:
            f.write(str(cm.mlir_module))


if __name__ == "__main__":
    main()
