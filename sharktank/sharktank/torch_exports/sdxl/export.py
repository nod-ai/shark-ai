# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

from iree.build import entrypoint, iree_build_main, cl_arg
from iree.turbine.aot import (
    ExportOutput,
    FxProgramsBuilder,
    export,
    externalize_module_parameters,
    save_module_parameters,
    decompositions,
)
from iree.turbine.aot.build_actions import turbine_generate

# USAGE
# python -m iree.build ./export.py --component=vae/scheduled_unet/clip --batch_size 1 ...


def create_safe_name(hf_model_name, model_name_str=""):
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub("\.", "_", safe_name)
    return safe_name


def export_sdxl_model(
    hf_model_name,
    component,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=64,
    external_weights=None,
    external_weights_path=None,
    decomp_attn=False,
    save_goldens=False,
    quant_paths=None,
) -> ExportOutput:
    import torch

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
            from clip import get_clip_model_and_inputs

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

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_clip_inputs,),
            )
            def encode_prompts(
                module,
                inputs,
            ):
                return module.forward(**inputs)

        elif component == "scheduled_unet":
            from unet import get_scheduled_unet_model_and_inputs

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
                external_weights_path,
                quant_paths,
            )
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_init_inputs,),
            )
            def init(
                module,
                inputs,
            ):
                return module.initialize(*inputs)

            @fxb.export_program(
                args=(sample_forward_inputs,),
            )
            def run_forward(
                module,
                inputs,
            ):
                return module.forward(*inputs)

        elif component == "vae":
            from vae import get_vae_model_and_inputs

            model, encode_args, decode_args = get_vae_model_and_inputs(
                hf_model_name, height, width, precision=precision, batch_size=batch_size
            )

            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(decode_args,),
            )
            def decode(
                module,
                inputs,
            ):
                return module.decode(**inputs)

        else:
            raise ValueError("Unimplemented: ", component)

    if external_weights:
        externalize_module_parameters(model)
        save_module_parameters(external_weights_path, model)
    module = export(fxb)
    return module


def get_filename(
    model,
    component,
    batch_size,
    max_length,
    height,
    width,
    precision,
):
    match component:
        case "scheduled_unet":
            return create_safe_name(
                model,
                f"scheduled_unet_bs{batch_size}_{max_length}_{height}x{width}_{precision}",
            )
        case "unet":
            return create_safe_name(
                model,
                f"unet_bs{batch_size}_{max_length}_{height}x{width}_{precision}",
            )
        case "clip":
            return create_safe_name(
                model,
                f"clip_bs{batch_size}_{max_length}_{precision}",
            )
        case "vae":
            return create_safe_name(
                model,
                f"vae_bs{batch_size}_{height}x{width}_{precision}",
            )
        case "scheduler":
            return create_safe_name(
                model,
                f"scheduler_bs{batch_size}_{height}x{width}_{precision}",
            )


@entrypoint(description="Builds artifacts for SDXL inference.")
def export_pipe(
    model=cl_arg(
        "model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HF model ID or path.",
    ),
    component=cl_arg(
        "component",
        default="scheduled_unet",
        help="Component to generate IR for. clip, vae, or scheduled_unet",
    ),
    batch_size=cl_arg(
        "batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
    ),
    height=cl_arg(
        "height",
        default=1024,
        help="Height of desired output image.",
    ),
    width=cl_arg(
        "width",
        default=1024,
        help="Width of desired output image.",
    ),
    precision=cl_arg(
        "precision",
        default="fp16",
        help="Model datatype. fp16, fp32, int8, fp8",
    ),
    max_length=cl_arg(
        "max_length",
        default=64,
        help="Max prompt tokens sequence length. Should be 64.",
    ),
    external_weights=cl_arg(
        "external_weights",
        default="irpa",
        help="Format for externalized weights. None inlines the weights, use irpa otherwise.",
    ),
    external_weights_path=cl_arg(
        "external_weights_path",
        default=None,
        help="Specify a non-default path for saved model parameters",
    ),
    decomp_attn=cl_arg(
        "decomp_attn",
        default=False,
        help="Explicitly decompose sdpa ops in the exported module.",
    ),
    quant_paths=cl_arg(
        "quant_paths", default=None, help="Path for quantized punet model artifacts."
    ),
):
    print(f"Export pid: {os.getpid()}")
    safe_name = get_filename(
        model,
        component,
        batch_size,
        max_length,
        height,
        width,
        precision,
    )
    if external_weights and not external_weights_path:
        external_weights_path = (
            create_safe_name(
                model,
                component + "_" + precision,
            )
            + "."
            + external_weights
        )

    module = turbine_generate(
        export_sdxl_model,
        hf_model_name=model,
        component=component,
        batch_size=batch_size,
        height=height,
        width=width,
        precision=precision,
        max_length=max_length,
        external_weights=external_weights,
        external_weights_path=external_weights_path,
        decomp_attn=decomp_attn,
        name=safe_name,
        out_of_process=False,
    )
    return f"{safe_name}.mlir"


if __name__ == "__main__":
    iree_build_main()
