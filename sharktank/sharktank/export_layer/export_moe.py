# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn.functional as F

from iree.turbine.aot import *

from amdsharktank.layers.mixture_of_experts_block import MoeBlock
from amdsharktank.utils import cli
from amdsharktank.types import Theta
from amdsharktank.layers.testing import make_random_moe_block_theta
from amdsharktank.utils.random import make_rand_torch


def main():
    parser = cli.create_parser()
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        default="/tmp/batch_llama_v1.mlir",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        help="Batch size to generate, e.g. `4` or `2`",
        type=lambda arg: int(arg),
        default="2",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Include verbose logging",
        action="store_true",
    )
    parser.add_argument(
        "--strict",
        help="Enables strictness during export",
        action="store_true",
    )
    parser.add_argument(
        "--use-gelu",
        help="Enable to use gelu for moe activation",
        action="store_true",
    )

    args = cli.parse(parser)

    bs = args.batch_size

    theta = Theta({"blk.0": make_random_moe_block_theta()})
    theta.rename_tensors_to_paths()
    model = MoeBlock(
        theta=theta("blk.0"),
        expert_count=8,
        expert_used_count=2,
        rms_epsilon=1e-5,
        moe_activation=F.gelu if args.use_gelu else F.silu,
    )
    fxb = FxProgramsBuilder(model)
    input = make_rand_torch((bs, 32, 6144))

    @fxb.export_program(name="prefill_moe", args=(input,))
    def _(model, input: torch.Tensor) -> torch.Tensor:
        return model(input)

    input = make_rand_torch((bs, 1, 6144))

    @fxb.export_program(name="decode_moe", args=(input,))
    def _(model, input: torch.Tensor) -> torch.Tensor:
        return model(input)

    if args.verbose:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)


if __name__ == "__main__":
    main()
