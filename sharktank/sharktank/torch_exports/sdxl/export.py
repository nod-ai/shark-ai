# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

from PIL import Image

from collections.abc import Iterable

from iree.compiler.ir import Context
from iree.turbine.aot import *
from iree.turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)
import torch
from safetensors import safe_open

from huggingface_hub import hf_hub_download

import numpy as np

from vae import get_vae_model_and_inputs
from clip import get_clip_model_and_inputs
from unet import get_scheduled_unet_model_and_inputs

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def flatten(xs):
    print("FLATTEN")
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes, torch.Tensor)):
            yield from flatten(x)
        else:
            yield x


def create_safe_name(hf_model_name, model_name_str=""):
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub("\.", "_", safe_name)
    return safe_name


@torch.no_grad()
def export_sdxl_model(
    hf_model_name,
    component,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=64,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    decomp_attn=False,
    save_goldens=True,
    quant_paths=None,
):
    dtype = torch_dtypes[precision]
    decomp_list = [torch.ops.aten.logspace]
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if component == "clip":
            model, sample_clip_inputs = get_clip_model_and_inputs(
                hf_model_name, max_length, precision, batch_size
            )
            if external_weights:
                # Transformers (model source) registers position ids as non-persistent.
                # This causes externalization to think it's a user input, and since it's not,
                # we end up trying to do ops on a !torch.None instead of a tensor.
                for buffer_name, buffer in model.named_buffers(recurse=True):
                    mod_name_list = buffer_name.split(".")
                    buffer_id = mod_name_list.pop()
                    parent = model
                    for i in mod_name_list:
                        parent = getattr(parent, i)
                    parent.register_buffer(buffer_id, buffer, persistent=True)
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)
            output = export(
                model,
                kwargs=sample_clip_inputs,
                module_name="compiled_clip",
                function_name="encode_prompts",
            )
            module = output.mlir_module
            sample_input_list = [sample_clip_inputs]
            golden_outputs = [model.forward(**sample_clip_inputs)]
        elif component == "scheduled_unet":
            (
                model,
                sample_init_inputs,
                sample_forward_inputs,
            ) = get_scheduled_unet_model_and_inputs(
                hf_model_name,
                height,
                width,
                max_length,
                precision,
                batch_size,
                external_weight_path,
                quant_paths,
            )
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_init_inputs,),
            )
            def _init(
                module,
                inputs,
            ):
                return module.initialize(*inputs)

            @fxb.export_program(
                args=(sample_forward_inputs,),
            )
            def _forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

            class CompiledScheduledUnet(CompiledModule):
                init = _init
                run_forward = _forward

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledScheduledUnet(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)
            sample_input_list = [sample_init_inputs, sample_forward_inputs]
            golden_outputs = []
        elif component == "vae":
            model, encode_args, decode_args = get_vae_model_and_inputs(
                hf_model_name, height, width, precision=precision, batch_size=batch_size
            )
            fxb = FxProgramsBuilder(model)

            # # TODO: fix issues with exporting the encode function.
            # @fxb.export_program(args=(encode_args,))
            # def _encode(module, inputs,):
            #     return module.encode(*inputs)

            @fxb.export_program(args=(decode_args,))
            def _decode(module, inputs):
                return module.decode(*inputs)

            class CompiledVae(CompiledModule):
                decode = _decode
                # encode = _encode

            if external_weights:
                externalize_module_parameters(model)
                save_module_parameters(external_weight_path, model)

            inst = CompiledVae(context=Context(), import_to="IMPORT")

            module = CompiledModule.get_mlir_module(inst)
            sample_input_list = [encode_args, decode_args]
            # golden_out_encode = model.encode(*encode_args)
            golden_out_decode = model.decode(*decode_args)
            golden_outputs = [golden_out_encode, golden_out_decode]
        else:
            raise ValueError("Unimplemented: ", component)
    if save_goldens:
        for idx, i in enumerate(sample_input_list):
            for num, inp in enumerate(i):
                np.save(f"{component}_{idx}_input_{num}.npy", inp)
        for idx, i in enumerate(golden_outputs):
            for num, out in enumerate(i):
                np.save(f"{component}_{num}_golden_out_{idx}.npy", np.asarray(out))
    module_str = str(module)
    return module_str


def get_filename(args):
    match args.component:
        case "scheduled_unet":
            return create_safe_name(
                args.model,
                f"scheduled_unet_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}",
            )
        case "unet":
            return create_safe_name(
                args.model,
                f"unet_bs{args.batch_size}_{args.max_length}_{args.height}x{args.width}_{args.precision}",
            )
        case "clip":
            return create_safe_name(
                args.model,
                f"clip_bs{args.batch_size}_{args.max_length}_{args.precision}",
            )
        case "vae":
            return create_safe_name(
                args.model,
                f"vae_bs{args.batch_size}_{args.height}x{args.width}_{args.precision}",
            )
        case "scheduler":
            return create_safe_name(
                args.model,
                f"scheduler_bs{args.batch_size}_{args.height}x{args.width}_{args.precision}",
            )


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(level=logging.DEBUG)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    p.add_argument(
        "--component",
        default="controlled_unet",
        choices=["unet", "scheduled_unet", "clip", "scheduler", "vae"],
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--precision", default="fp16")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--external_weights", default="irpa")
    p.add_argument("--external_weights_path", default=None)
    p.add_argument("--decomp_attn", action="store_true")
    args = p.parse_args()

    if args.external_weights and not args.external_weights_path:
        args.external_weights_path = (
            create_safe_name(
                args.model,
                args.component + "_" + args.precision,
            )
            + "."
            + args.external_weights
        )
    safe_name = get_filename(args)
    mod_str = export_sdxl_model(
        args.model,
        args.component,
        args.batch_size,
        args.height,
        args.width,
        args.precision,
        args.max_length,
        "mlir",
        args.external_weights,
        args.external_weights_path,
        args.decomp_attn,
    )

    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
