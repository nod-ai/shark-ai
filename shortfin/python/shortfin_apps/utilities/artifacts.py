# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import os
import itertools
import shortfin.array as sfnp

dtype_to_filetag = {
    sfnp.float16: "fp16",
    sfnp.float32: "fp32",
    sfnp.int8: "i8",
    sfnp.bfloat16: "bf16",
}


def fetch_modules(
    target,
    device,
    model_id,
    model_config,
    artifacts_dir=None,
    splat=False,
):
    if model_id == "sdxl":
        from shortfin_apps.sd.components.config_struct import ModelParams
    elif model_id == "flux":
        from shortfin_apps.flux.components.config_struct import ModelParams
    else:
        raise ValueError(f"Model Params not defined for: {model_id}")

    mod_params = ModelParams.load_json(model_config)

    vmfbs = {}
    params = {}
    for submodel in mod_params.module_names.keys():
        vmfbs[submodel] = {}
        for bs in mod_params.batch_sizes[submodel]:
            vmfbs[submodel][bs] = []
        if submodel != "scheduler":
            params[submodel] = []

    cached, missing = fetch_all_modules(
        mod_params, target, device, model_id, splat, artifacts_dir
    )

    for name in cached:
        if "vmfb" in name:
            for key in vmfbs.keys():
                for bs in vmfbs[key].keys():
                    if key in name.lower() and f"_bs{bs}_" in name.lower():
                        vmfbs[key][bs].extend([name])
        elif "irpa" in name:
            for key in params.keys():
                if key in name.lower():
                    params[key] = [name]

    return vmfbs, params, missing


def fetch_all_modules(
    model_params, target: str, driver: str, model: str, splat: bool, artifacts_dir: str
) -> (list, list):
    required_vmfbs = get_vmfb_filenames(
        model_params,
        None,
        target,
        driver,
    )
    required_params = get_params_filenames(model_params, None, splat)

    vmfb_filepaths = [
        os.path.join(artifacts_dir, "bin", model, vmfb_name)
        for vmfb_name in required_vmfbs
    ]
    param_filepaths = [
        os.path.join(artifacts_dir, "genfiles", model, params_name)
        for params_name in required_params
    ]

    all_paths = [*vmfb_filepaths, *param_filepaths]
    cached = []
    missing = []
    for path in all_paths:
        if os.path.exists(path):
            cached.extend([path])
        else:
            missing.extend([path])
    return cached, missing


def get_vmfb_filenames(
    model_params,
    model=None,
    target: str = "gfx942",
    driver: str = "amdgpu",
) -> list:
    vmfb_filenames = []
    file_stems = get_file_stems(model_params)
    for stem in file_stems:
        vmfb_filenames.extend([stem + "_" + target + "_" + driver + ".vmfb"])
    return vmfb_filenames


def get_params_filenames(model_params, model=None, splat: bool = False):
    # TODO: Make this a service-agnostic common utility.

    base = create_safe_name(model_params.base_model_name)

    modnames = ["clip", "vae"]
    mod_precisions = [
        dtype_to_filetag[model_params.clip_dtype],
        dtype_to_filetag[model_params.unet_dtype],
    ]
    if model_params.use_punet:
        modnames.append("punet")
        mod_precisions.append(model_params.unet_quant_dtype)
    else:
        modnames.append("unet")
        mod_precisions.append(dtype_to_filetag[model_params.unet_dtype])

    params_filenames = []
    if splat == "True":
        for idx, mod in enumerate(modnames):
            params_filenames.extend(
                ["_".join([mod, "splat", f"{mod_precisions[idx]}.irpa"])]
            )
    else:
        for idx, mod in enumerate(modnames):
            params_filenames.extend(
                [base + "_" + mod + "_dataset_" + mod_precisions[idx] + ".irpa"]
            )
    if model:
        filenames = filter_by_model(params_filenames, model)
    else:
        filenames = params_filenames

    return filenames


def create_safe_name(hf_model_name, model_name_str="") -> str:
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub(r"\.", "_", safe_name)
    return safe_name


def get_file_stems(model_params) -> list[str]:
    # TODO: Make this a service-agnostic common utility.
    file_stems = []

    # Create a dictionary of service components and their filename UIDs.
    mod_names = {
        "clip": "clip",
        "vae": "vae",
    }
    if model_params.use_scheduled_unet:
        mod_names.update(
            {
                "scheduled_unet": "scheduled_unet",
            }
        )
    else:
        mod_names.update(
            {
                "unet": "punet",
                "scheduler": model_params.scheduler_id + "Scheduler",
            }
        )

    base = create_safe_name(model_params.base_model_name)

    for mod, modname in mod_names.items():
        # Given parametrizations from model config, compile an exhaustive list of unique module file stems matching the configuration.
        ord_params = [
            [base],
            [modname],
        ]

        # Batch sizes.
        bsizes = []
        for bs in model_params.batch_sizes[mod]:
            bsizes.extend([f"bs{bs}"])
        ord_params.extend([bsizes])

        # Sequence length.
        if mod in ["scheduled_unet", "unet", "clip"]:
            ord_params.extend([[str(model_params.max_seq_len)]])

        # Output image dims.
        if mod in ["scheduled_unet", "unet", "vae", "scheduler"]:
            dims = []
            for dim_pair in model_params.dims:
                dim_pair_str = [str(d) for d in dim_pair]
                dims.extend(["x".join(dim_pair_str)])
            ord_params.extend([dims])

        # Precision.
        if mod == "scheduler":
            dtype_str = dtype_to_filetag[model_params.unet_dtype]
        elif "unet" not in modname:
            dtype_str = dtype_to_filetag[
                getattr(model_params, f"{mod}_dtype", sfnp.float16)
            ]
        else:
            dtype_str = model_params.unet_quant_dtype
        ord_params.extend([[dtype_str]])

        # Generate file stems.
        for x in list(itertools.product(*ord_params)):
            file_stems.extend(["_".join(x)])

    return file_stems
