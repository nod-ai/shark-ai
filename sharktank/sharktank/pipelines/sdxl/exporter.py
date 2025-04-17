# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import argparse
from copy import copy
import functools
import logging
from typing import Any, Dict, Optional, Set, Union
from pathlib import Path
from dataclasses import fields

from iree.turbine.aot import (
    ExportOutput,
    FxProgramsBuilder,
    export,
    externalize_module_parameters,
    save_module_parameters,
    decompositions,
)

from sharktank.models.vae.model import VaeDecoderModel
from sharktank.models.clip import ClipTextModel, ClipTextConfig
from sharktank.tools.import_hf_dataset import import_hf_dataset

def export_sdxl_model(
    hf_model_name,
    component,
    batch_size,
    height,
    width,
    precision="fp16",
    max_length=64,
    external_weights=None,
    external_weights_file=None,
    decomp_attn=False,
    quant_path=None,
    scheduler_config_path=None,
    weights_only=False,
) -> ExportOutput:
    import torch

    def check_torch_version(begin: tuple, end: tuple):
        pass

    decomp_list = [torch.ops.aten.logspace]
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if component == "clip":
            from sharktank.pipelines.sdxl.clip import get_clip_model_and_inputs

            module_name = "compiled_clip"
            model, sample_clip_inputs = get_clip_model_and_inputs(
                hf_model_name, max_length, precision, batch_size
            )
            # if external_weights:
            #     # Transformers (model source) registers position ids as non-persistent.
            #     # This causes externalization to think it's a user input, and since it's not,
            #     # we end up trying to do ops on a !torch.None instead of a tensor.
            #     for buffer_name, buffer in model.named_buffers(recurse=True):
            #         mod_name_list = buffer_name.split(".")
            #         buffer_id = mod_name_list.pop()
            #         parent = model
            #         for i in mod_name_list:
            #             parent = getattr(parent, i)
            #         parent.register_buffer(buffer_id, buffer, persistent=True)
            # model.to("cpu")
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(sample_clip_inputs,),
            )
            def encode_prompts(
                module,
                inputs,
            ):
                return module.forward(**inputs)

        elif component in ["unet", "punet", "scheduled_unet"]:
            check_torch_version((2, 4, 1), (2, 6, 0))
            from sharktank.pipelines.sdxl.unet import (
                get_scheduled_unet_model_and_inputs,
                get_punet_model_and_inputs,
            )

            if component in ["unet", "punet"]:
                module_name = "compiled_punet"
                implementation = get_punet_model_and_inputs
            else:
                module_name = "compiled_spunet"
                implementation = get_scheduled_unet_model_and_inputs
            (model, sample_init_inputs, sample_forward_inputs,) = implementation(
                hf_model_name,
                height,
                width,
                max_length,
                precision,
                batch_size,
                external_weights_file,
                quant_path,
                scheduler_config_path,
            )
            if external_weights:
                externalize_module_parameters(model.cond_model)
            if component == "scheduled_unet":
                fxb = FxProgramsBuilder(model)

                @fxb.export_program(
                    args=(sample_init_inputs,),
                )
                def run_initialize(
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

                return export(fxb, module_name=module_name)
            else:
                return export(
                    model, kwargs=sample_forward_inputs, module_name="compiled_punet"
                )
        elif component == "scheduler":
            module_name = "compiled_scheduler"
            from sharktank.pipelines.sdxl.scheduler import (
                get_scheduler_model_and_inputs,
            )

            model, init_args, prep_args, step_args = get_scheduler_model_and_inputs(
                hf_model_name if not scheduler_config_path else scheduler_config_path,
                batch_size,
                height,
                width,
                precision,
            )
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(init_args,),
            )
            def run_initialize(module, sample):
                return module.initialize(*sample)

            @fxb.export_program(
                args=(prep_args,),
            )
            def run_scale(module, inputs):
                return module.scale_model_input(*inputs)

            @fxb.export_program(
                args=(step_args,),
            )
            def run_step(module, inputs):
                return module.step(*inputs)

        elif component == "vae":
            from sharktank.pipelines.sdxl.vae import get_vae_model_and_inputs

            module_name = "compiled_vae"
            if quant_path and os.path.exists(
                os.path.join(quant_path, "vae.safetensors")
            ):
                vae_path = os.path.join(quant_path, "vae.safetensors")
            else:
                vae_path = None
            model, encode_args, decode_args = get_vae_model_and_inputs(
                hf_model_name,
                height,
                width,
                precision=precision,
                batch_size=batch_size,
                custom_vae_path=vae_path,
            )
            fxb = FxProgramsBuilder(model)

            @fxb.export_program(
                args=(decode_args,),
            )
            def decode(
                module,
                inputs,
            ):
                return module.decode(*inputs)

        else:
            raise ValueError("Unimplemented: ", component)

    if external_weights:
        externalize_module_parameters(model)

    module = export(fxb, module_name=module_name)
    return module

torch_dtypes = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def is_already_exported(output_path: Path) -> bool:
    return output_path.exists()


def find_safetensors_files(path: Path) -> list[Path]:
    """Find all .safetensors files in a directory, excluding index files."""
    safetensors_files = list(path.glob("*.safetensors"))
    safetensors_files.sort()
    return safetensors_files


def filter_properties_for_config(
    properties: Dict[str, Any], config_class: Any
) -> Dict[str, Any]:
    """Filter properties to only include fields valid for the given config class.

    Args:
        properties: Properties dictionary
        config_class: The dataclass to filter properties for

    Returns:
        Filtered properties dictionary with only valid fields for the config class
    """
    # Start with hparams if available
    if "hparams" in properties:
        props = properties["hparams"]
    else:
        props = properties

    # Get set of valid field names for the config class
    valid_fields = {f.name for f in fields(config_class)}

    # Filter to only include valid fields
    filtered_props = {k: v for k, v in props.items() if k in valid_fields}

    return filtered_props

def export_sdxl_submodel_parameters(
    submodel: str,
    source_path: str,
    output_path: str,
    source_subfolder: Optional[str] = None,
    params_filename: Optional[str] = None,
    dtype_str: Optional[str] = None,
)
    """Export SDXL pipeline submodel parameters to IREE format.

    Args:
        submodel: "clip", "punet", "vae". 
        source_path: Path to model files. Can be a local directory or huggingface repo ID.
        output_path: Output path for IREE parameters
        source_subfolder: Specify a source subfolder from which to pull a submodel's config and weights.
    """
    # Ensure output_path is a Path object
    if not model_name:
        model_name = "stable_diffusion_xl_base_1_0"
    if not dtype_str:
        dtype_str = "fp16"
    if not params_filename:
        params_filename = "model.safetensors"
    dtype = torch_dtypes[dtype_str]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_src = source_path + "/" + source_subfolder if source_subfolder is not None else source_path

    # Export parameters
    if submodel == "punet":
        from sharktank.models.punet.tools import import_brevitas_dataset
        
        config_json_path = model_source / "config.json"
        params_path = model_source / params_filename
        quant_params_path = model_source / "quant_params.json"
        ds_import_args = [
            f"--config-json={config_json_path}",
            f"--params={params_path}",
            f"--output-irpa-file={output_path}",
        ]
        if quant_params_path:
            ds_import_args.extend([f"--quant-params={quant_params_path}"])
        import_brevitas_dataset.main(ds_import_args)
        
    elif not is_already_exported(output_path):
        config_json_path = clip_path / "config.json"
        param_paths = find_safetensors_files(clip_path)
        clip_dataset = import_hf_dataset(
            config_json_path, param_paths, target_dtype=dtype
        )
        clip_dataset.properties = filter_properties_for_config(
            clip_dataset.properties, ClipTextConfig
        )
        clip_dataset.save(str(clip_output_path))
        logging.info(f"Exported CLIP parameters to {clip_output_path}")
    else:
        logging.info(
            f"Skipped CLIP parameter export, already exists at {clip_output_path}"
        )

    # Export VAE parameters
    vae_path = Path(model_path) / "vae/"
    vae_output_path = output_path / f"{model_name.split('_')[0]}_vae_{dtype_str}.irpa"
    if not is_already_exported(vae_output_path):
        config_json_path = vae_path / "config.json"
        param_paths = find_safetensors_files(vae_path)
        import_hf_dataset(
            config_json_path, param_paths, vae_output_path, target_dtype=dtype
        )
        logging.info(f"Exported VAE parameters to {vae_output_path}")
    else:
        logging.info(
            f"Skipped VAE parameter export, already exists at {vae_output_path}"
        )

    logging.info(f"Completed Flux pipeline parameter export to {output_path}")

def export_sdxl_pipeline_iree_parameters(
    pipeline_config,
    output_path,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe_config_path", type=str, default="./configs/sdxl_config_i8.json")
    parser.add_argument("--output-path", type=str, default="~/.cache/shark/genfiles/sdxl/")
    parser.add_argument("--export", type=str, choices=["irpa", "mlir", "all"], required=True, help="Export target.")
    args = parser.parse_args()

    exported = export_sdxl_pipeline_iree_parameters(
        args.pipe_config_path,
        args.output_path,
    )
    print(exported)

    if args.export in ["mlir", "all"]:
        mlir_outputs = export_sdxl_pipeline_mlir(exported, args.output_path)
        print(mlir_outputs)


if __name__ == "__main__":
    main()
