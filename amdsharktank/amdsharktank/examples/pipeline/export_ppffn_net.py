# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Example program to export a sharded and pipeline parallized set FFN networks.
This is used for developing and testing various tooling flows with a scaled down example.

Generate MLIR and a random inited IRPA file with:

    python -m amdsharktank.examples.sharding.export_pffn_net \
        --output-irpa-file=/tmp/ffn.irpa /tmp/ffn.mlir
"""

import os
import math

import torch

from amdsharktank.utils import cli
from amdsharktank.layers import *
from amdsharktank import ops
from amdsharktank.types import *
from amdsharktank.types.pipelining import (
    pipeline_parallelize_llm_theta,
    transfer_between_blocks,
)

from iree.turbine.aot import DeviceAffinity, export


def create_theta(
    dim: int, tensor_parallelism_size: int, blocks_count: int, save_path: str
) -> None:
    """
    Create the IRPA file for the example and save it to `save_path`.

    Args:
        dim: Dimension of the square FFN layer weights.
        tensor_parallelism_size: Number of shards to split the weights into.
        blocks_count: Number of FFN blocks to create.
        save_path: Path to save the IRPA file.
    """
    split_size = dim // tensor_parallelism_size
    weights = []
    for layer in range(blocks_count):
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

    # Note: Next three weights are unused in the example but are expected by `pipeline_parallelize_llm_theta`.
    ones = torch.ones(dim, dim, dtype=torch.float16)
    weights.append(
        SplitPrimitiveTensor(
            name="token_embd.weight",
            shard_dim=1,
            ts=ones.split(split_size, dim=1),
        )
        if tensor_parallelism_size > 1
        else DefaultPrimitiveTensor(name="token_embd.weight", data=ones)
    )
    weights.append(
        SplitPrimitiveTensor(
            name="output.weight",
            shard_dim=1,
            ts=ones.split(split_size, dim=1),
        )
        if tensor_parallelism_size > 1
        else DefaultPrimitiveTensor(name="output.weight", data=ones)
    )
    ones = torch.ones(1, dim, dtype=torch.float16)
    weights.append(
        SplitPrimitiveTensor(
            name="output_norm.weight",
            shard_dim=1,
            ts=ones.split(split_size, dim=1),
        )
        if tensor_parallelism_size > 1
        else DefaultPrimitiveTensor(name="output_norm.weight", data=ones)
    )

    ds = Dataset({}, Theta(weights))
    ds.save(save_path)


class PPFFN(ThetaLayer):
    def __init__(self, theta):
        super().__init__(theta)
        self.blocks = torch.nn.ModuleList(
            LinearLayer(theta(f"blk.{block_idx}.ffn"))
            for block_idx in range(len(theta.tensor("blk")))
        )

    def forward(self, x: torch.Tensor):
        for block_idx, block in enumerate(self.blocks):
            x = transfer_between_blocks(
                x, curr_block_tensors=self.theta.tensor("blk", block_idx)
            )
            x = block(x)

        return ops.unshard(x)


def main(raw_args=None):
    parser = cli.create_parser()
    parser.add_argument(
        "--tensor-parallelism-size",
        type=int,
        default=1,
        help="Number of shards to split a tensor into.",
    )
    parser.add_argument(
        "--pipeline-parallelism-size",
        type=int,
        default=2,
        help="Number of pipeline stages.",
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
    assert (
        args.pipeline_parallelism_size > 1
    ), "Pipeline parallelism size must be greater than 1."

    bs = 16
    sl = 128
    dim = 128 * 2**5
    block_count = 24
    create_theta(
        dim, args.tensor_parallelism_size, block_count, save_path=args.output_irpa_file
    )

    parallelism_config = ParallelismConfig.default_config(
        block_count=block_count, pp=args.pipeline_parallelism_size
    )

    ds = Dataset.load(args.output_irpa_file)
    pipeline_parallelize_llm_theta(ds.root_theta, parallelism_config)

    mdl = PPFFN(ds.root_theta)

    example_arg = torch.empty(bs, sl, dim, dtype=torch.float16)
    arg_devices = {0: parallelism_config.device_affinity_for_pipeline(0)}

    ep = torch.export.export(mdl, (example_arg,), strict=False)
    cm = export(ep, arg_device=arg_devices)

    if args.output_file == "-":
        print(cm.mlir_module)
    else:
        with open(args.output_file, "wt") as f:
            f.write(str(cm.mlir_module))


if __name__ == "__main__":
    main()
