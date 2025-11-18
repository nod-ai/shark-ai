# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import sys

import torch

from iree.turbine import aot

from ..model import Unet2DConditionModel, ClassifierFreeGuidanceUnetModel
from amdsharktank.utils.patching import SaveModuleResultTensorsPatch
from amdsharktank.utils import cli
from .sample_data import get_random_inputs, load_inputs, save_outputs


def main(argv):

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    parser.add_argument("--device", default="cuda:0", help="Torch device to run on")
    parser.add_argument("--dtype", default="float16", help="DType to run in")
    parser.add_argument("--export", type=Path, help="Export to path (vs run)")
    parser.add_argument("--bs", default=1, type=int, help="Batch size for export")
    parser.add_argument(
        "--inputs",
        type=Path,
        help="Safetensors file of inputs (or random if not given)",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        help="Safetensors file of outputs",
    )
    parser.add_argument(
        "--save-intermediates-path",
        type=Path,
        help="Path of safetensors file in which to save all module outputs",
    )
    args = cli.parse(parser, args=argv)

    device = args.device
    dtype = getattr(torch, args.dtype)

    ds = cli.get_input_dataset(args)
    ds.to(device=device)

    cond_unet = Unet2DConditionModel.from_dataset(ds)
    mdl = ClassifierFreeGuidanceUnetModel(cond_unet)

    # Run a step for debugging.
    if args.inputs:
        inputs = load_inputs(args.inputs, dtype=dtype, device=device, bs=args.bs)
    else:
        inputs = get_random_inputs(dtype=dtype, device=device, bs=args.bs)

    if args.export:
        # Temporary: Need a dedicated exporter.
        output = aot.export(
            mdl,
            kwargs=inputs,
        )
        output.save_mlir(args.export)
    else:
        # Save intermediates.
        intermediates_saver = None
        if args.save_intermediates_path:
            intermediates_saver = SaveModuleResultTensorsPatch()
            intermediates_saver.patch_child_modules(mdl.cond_model)

        results = mdl.forward(**inputs)
        print("1-step results:", results)
        if args.outputs:
            print(f"Saving outputs to {args.outputs}")
            save_outputs(args.outputs, results)

        if intermediates_saver:
            print(f"Saving intermediate tensors to: {args.save_intermediates_path}")
            intermediates_saver.save_file(args.save_intermediates_path)


if __name__ == "__main__":
    main(sys.argv[1:])
