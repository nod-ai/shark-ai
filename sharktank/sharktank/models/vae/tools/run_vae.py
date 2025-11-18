# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import sys

import torch

from iree.turbine import aot

from amdsharktank.layers import ThetaLayer
from amdsharktank.utils.patching import SaveModuleResultTensorsPatch
from amdsharktank.utils import cli
from amdsharktank.models.punet.tools.sample_data import load_inputs, save_outputs
from .sample_data import get_random_inputs
from ..model import VaeDecoderModel
from iree.turbine.aot import FxProgramsBuilder, export, decompositions

from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
import numpy as np


def export_vae(model: ThetaLayer, sample_inputs, decomp_attn: bool) -> aot.ExportOutput:
    for t in model.theta.flatten().values():
        aot.ExternalTensorTrait(external_name=t.name, external_scope="").set(
            t.as_torch()
        )

    decomp_list = []
    if decomp_attn:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True, add_ops=decomp_list
    ):

        fxb = FxProgramsBuilder(model)

        @fxb.export_program(
            name=f"decode",
            args=tuple(torch.unsqueeze(sample_inputs, 0)),
            strict=False,
        )
        def _(
            model,
            sample_inputs,
        ):
            return model(sample_inputs)

        output = export(fxb, import_symbolic_shape_expressions=True)
        return output


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
    parser.add_argument(
        "--compare_vs_torch",
        action="store_true",
        help="Compares results vs HF diffusers reference model",
    )

    parser.add_argument(
        "--torch_model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HF reference model id, currently tested with stabilityai/stable-diffusion-xl-base-1.0 and black-forest-labs/FLUX.1-dev",
    )

    parser.add_argument(
        "--amdsharktank_config",
        default="sdxl",
        help="amdsharktank config providing hyperparamters [sdxl or flux]",
    )

    parser.add_argument(
        "--decomp_attn",
        action="store_true",
        help="Decomposes the attention op during export",
    )
    args = cli.parse(parser, args=argv)

    device = args.device
    dtype = getattr(torch, args.dtype)

    ds = cli.get_input_dataset(args)
    ds.to(device=device)

    mdl = VaeDecoderModel.from_dataset(ds)
    # Run a step for debugging.
    if args.inputs:
        inputs = load_inputs(args.inputs, dtype=dtype, device=device, bs=args.bs)
    else:
        inputs = get_random_inputs(
            dtype=dtype, device=device, bs=args.bs, config=args.amdsharktank_config
        )

    if args.export:
        # TODO move export from a run_vae file
        output = export_vae(mdl, inputs, args.decomp_attn)
        output.save_mlir(args.export)
        print("exported VAE model. Skipping eager execution")
    else:
        # Save intermediates.
        intermediates_saver = None
        if args.save_intermediates_path:
            intermediates_saver = SaveModuleResultTensorsPatch()
            intermediates_saver.patch_child_modules(mdl.cond_model)

        results = mdl.forward(inputs)
        print("results:", results)

        if args.outputs:
            print(f"Saving outputs to {args.outputs}")
            save_outputs(args.outputs, results)

        if intermediates_saver:
            print(f"Saving intermediate tensors to: {args.save_intermediates_path}")
            intermediates_saver.save_file(args.save_intermediates_path)

        if args.compare_vs_torch:
            from .diffuser_ref import run_torch_vae

            if args.amdsharktank_config == "flux":
                diffusers_results = run_torch_vae(
                    args.torch_model, inputs, flux=True, dtype=dtype
                )
            elif args.amdsharktank_config == "sdxl":
                run_torch_vae(args.torch_model, inputs, flux=False, dtype=dtype)
            print("diffusers results:", diffusers_results)
            torch.testing.assert_close(diffusers_results, results)


if __name__ == "__main__":
    main(sys.argv[1:])
