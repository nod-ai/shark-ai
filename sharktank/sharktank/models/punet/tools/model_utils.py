# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import torch

from sharktank.types.theta import load_properties
from sharktank.utils import tree, parse_version
from sharktank.models.punet.model import Unet2DConditionModel, ClassifierFreeGuidanceUnetModel, ClassifierFreeGuidanceScheduledUnetModel

def get_punet_module_variant(
    repo_id: str, 
    subfolder: str,
    revision: str, 
    variant: str, 
    batch_size: int,
    height: int,
    width: int,
    max_length: int,
    precision: str,
    external_weight_path: str,
    quant_path: str = None,
    scheduler_config_path: str = None,
    ):
    hf_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    if quant_path is not None and os.path.exists(quant_path):
        quant_paths = {
            "config": f"{quant_path}/config.json",
            "params": f"{quant_path}/params.safetensors",
            "quant_params": f"{quant_path}/quant_params.json",
        }
    else:
        quant_paths = None
    if precision == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16
    punet = get_punet_model(
        hf_model_name,
        external_weight_path,
        quant_paths,
        precision,
        repo_id,
        subfolder,
        revision,
    )
    if variant == "scheduled":
        from sharktank.models.punet.scheduler import get_scheduler
        if not scheduler_config_path:
            scheduler_config_path = hf_model_name
        raw_scheduler = get_scheduler(scheduler_config_path, "EulerDiscrete")
        return ClassifierFreeGuidanceScheduledUnetModel(
            punet,
            raw_scheduler,
            height,
            width,
            max_length,
            batch_size,
            dtype
        )
    else:
        return punet


def get_punet_model(hf_model_name, external_weight_path, quant_paths, precision="i8", repo_id=None, subfolder=None, revision=None):
    from huggingface_hub import hf_hub_download

    if os.path.exists(external_weight_path):
        return Unet2DConditionModel.from_irpa(external_weight_path)

    if repo_id and subfolder and revision:
        repo_id = repo_id
        subfolder = subfolder
        revision = revision
    elif precision in ["fp8", "f8"]:
        repo_id = "amd-shark/sdxl-quant-models"
        subfolder = "unet/int8"
        revision = "a31d1b1cba96f0da388da348bcaee197a073d451"
    elif precision == "fp8_ocp":
        repo_id = "amd-shark/sdxl-quant-fp8"
        subfolder = "unet_int8_sdpa_fp8_ocp"
        revision = "e6e3c031e6598665ca317b80c3b627c186ca08e7"
    else:
        repo_id = "amd-shark/sdxl-quant-int8"
        subfolder = "mi300_all_sym_8_step14_fp32"
        revision = "efda8afb35fd72c1769e02370b320b1011622958"

    # TODO (monorimet): work through issues with pure fp16 punet export. Currently int8 with fp8/fp8_ocp/fp16 sdpa are supported.

    def download(filename):
        return hf_hub_download(
            repo_id=repo_id, subfolder=subfolder, filename=filename, revision=revision
        )

    if quant_paths and quant_paths["config"] and os.path.exists(quant_paths["config"]):
        results = {
            "config.json": quant_paths["config"],
        }
    else:
        try:
            results = {
                "config.json": download("config.json"),
            }
        except:
            # Fallback to original model config file.
            results = {
                "config.json": hf_hub_download(
                    repo_id=hf_model_name,
                    subfolder="unet",
                    filename="config.json",
                )
            }
    if quant_paths and quant_paths["params"] and os.path.exists(quant_paths["params"]):
        results["params.safetensors"] = quant_paths["params"]
    else:
        results["params.safetensors"] = download("params.safetensors")

    output_dir = os.path.dirname(external_weight_path)

    if (
        quant_paths
        and quant_paths["quant_params"]
        and os.path.exists(quant_paths["quant_params"])
    ):
        results["quant_params.json"] = quant_paths["quant_params"]
    else:
        results["quant_params.json"] = download("quant_params.json")
    ds_filename = os.path.basename(external_weight_path)
    output_path = os.path.join(output_dir, ds_filename)
    ds = get_punet_dataset(
        results["config.json"],
        results["params.safetensors"],
        output_path,
        results["quant_params.json"],
    )

    cond_unet = Unet2DConditionModel.from_dataset(ds)
    return cond_unet


def get_punet_dataset(
    config_json_path,
    params_path,
    output_path,
    quant_params_path=None,
):
    from sharktank.models.punet.tools import import_brevitas_dataset

    ds_import_args = [
        f"--config-json={config_json_path}",
        f"--params={params_path}",
        f"--output-irpa-file={output_path}",
    ]
    if quant_params_path:
        ds_import_args.extend([f"--quant-params={quant_params_path}"])
    import_brevitas_dataset.main(ds_import_args)
    return import_brevitas_dataset.Dataset.load(output_path)
