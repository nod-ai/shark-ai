# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.build import *
import itertools
import os
import shortfin.array as sfnp
import copy

from shortfin_apps.sd.components.config_struct import ModelParams
from shortfin_apps.utils import *

this_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(this_dir)
default_config_json = os.path.join(parent, "examples", "sdxl_config_i8.json")


ARTIFACT_VERSION = "11182024"
SDXL_BUCKET = (
    f"https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/{ARTIFACT_VERSION}/"
)
SDXL_WEIGHTS_BUCKET = (
    "https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/weights/"
)


def filter_by_model(filenames, model):
    if not model:
        return filenames
    filtered = []
    for i in filenames:
        if model.lower() in i.lower():
            filtered.extend([i])
    return filtered


def get_mlir_filenames(model_params: ModelParams, model=None):
    mlir_filenames = []
    file_stems = get_file_stems(model_params)
    for stem in file_stems:
        mlir_filenames.extend([stem + ".mlir"])
    return filter_by_model(mlir_filenames, model)


def get_vmfb_filenames(
    model_params: ModelParams, model=None, target: str = "amdgpu-gfx942"
):
    vmfb_filenames = []
    file_stems = get_file_stems(model_params)
    for stem in file_stems:
        vmfb_filenames.extend([stem + "_" + target + ".vmfb"])
    return filter_by_model(vmfb_filenames, model)


def get_params_filenames(model_params: ModelParams, model=None, splat: bool = False):
    params_filenames = []
    base = (
        "stable_diffusion_xl_base_1_0"
        if model_params.base_model_name.lower() == "sdxl"
        else model_params.base_model_name
    )
    modnames = ["clip", "vae"]
    mod_precs = [
        dtype_to_filetag[model_params.clip_dtype],
        dtype_to_filetag[model_params.unet_dtype],
    ]
    if model_params.use_i8_punet:
        modnames.append("punet")
        mod_precs.append("i8")
    else:
        modnames.append("unet")
        mod_precs.append(dtype_to_filetag[model_params.unet_dtype])
    if splat == "True":
        for idx, mod in enumerate(modnames):
            params_filenames.extend(
                ["_".join([mod, "splat", f"{mod_precs[idx]}.irpa"])]
            )
    else:
        for idx, mod in enumerate(modnames):
            params_filenames.extend(
                [base + "_" + mod + "_dataset_" + mod_precs[idx] + ".irpa"]
            )
    return filter_by_model(params_filenames, model)


def get_file_stems(model_params: ModelParams):
    file_stems = []
    base = (
        ["stable_diffusion_xl_base_1_0"]
        if model_params.base_model_name.lower() == "sdxl"
        else [model_params.base_model_name]
    )
    mod_names = {
        "clip": "clip",
        "unet": "punet" if model_params.use_i8_punet else "unet",
        "scheduler": model_params.scheduler_id + "Scheduler",
        "vae": "vae",
    }
    for mod, modname in mod_names.items():
        ord_params = [
            base,
            [modname],
        ]
        bsizes = []
        for bs in getattr(model_params, f"{mod}_batch_sizes", [1]):
            bsizes.extend([f"bs{bs}"])
        ord_params.extend([bsizes])
        if mod in ["unet", "clip"]:
            ord_params.extend([[str(model_params.max_seq_len)]])
        if mod in ["unet", "vae", "scheduler"]:
            dims = []
            for dim_pair in model_params.dims:
                dim_pair_str = [str(d) for d in dim_pair]
                dims.extend(["x".join(dim_pair_str)])
            ord_params.extend([dims])
        if mod == "scheduler":
            dtype_str = dtype_to_filetag[model_params.unet_dtype]
        elif mod != "unet":
            dtype_str = dtype_to_filetag[
                getattr(model_params, f"{mod}_dtype", sfnp.float16)
            ]
        else:
            dtype_str = (
                "i8"
                if model_params.use_i8_punet
                else dtype_to_filetag[model_params.unet_dtype]
            )
        ord_params.extend([[dtype_str]])
        for x in list(itertools.product(*ord_params)):
            file_stems.extend(["_".join(x)])
    return file_stems


@entrypoint(description="Retreives a set of SDXL submodels.")
def sdxl(
    model_json=cl_arg(
        "model-json",
        default=default_config_json,
        help="Local config filepath",
    ),
    target=cl_arg(
        "target",
        default="gfx942",
        help="IREE target architecture.",
    ),
    splat=cl_arg(
        "splat", default=False, type=str, help="Download empty weights (for testing)"
    ),
    build_preference=cl_arg(
        "build-preference",
        default="precompiled",
        help="Sets preference for artifact generation method: [compile, precompiled]",
    ),
    model=cl_arg("model", type=str, help="Submodel to fetch/compile for."),
):
    model_params = ModelParams.load_json(model_json)
    ctx = executor.BuildContext.current()
    update = needs_update(ctx, ARTIFACT_VERSION)

    mlir_bucket = SDXL_BUCKET + "mlir/"
    vmfb_bucket = SDXL_BUCKET + "vmfbs/"
    if "gfx" in target:
        target = "amdgpu-" + target

    mlir_filenames = get_mlir_filenames(model_params, model)
    mlir_urls = get_url_map(mlir_filenames, mlir_bucket)
    for f, url in mlir_urls.items():
        if update or needs_file_url(f, ctx, url):
            fetch_http(name=f, url=url)

    vmfb_filenames = get_vmfb_filenames(model_params, model=model, target=target)
    vmfb_urls = get_url_map(vmfb_filenames, vmfb_bucket)
    if build_preference == "compile":
        for idx, f in enumerate(copy.deepcopy(vmfb_filenames)):
            # We return .vmfb file stems for the compile builder.
            file_stem = "_".join(f.split("_")[:-1])
            if needs_compile(file_stem, target, ctx):
                for mlirname in mlir_filenames:
                    if file_stem in mlirname:
                        mlir_source = mlirname
                        break
                obj = compile(name=file_stem, source=mlir_source)
                vmfb_filenames[idx] = obj[0]
            else:
                vmfb_filenames[idx] = get_cached_vmfb(file_stem, target, ctx)
    else:
        for f, url in vmfb_urls.items():
            if update or needs_file_url(f, ctx, url):
                fetch_http(name=f, url=url)

    params_filenames = get_params_filenames(model_params, model=model, splat=splat)
    params_urls = get_url_map(params_filenames, SDXL_WEIGHTS_BUCKET)
    for f, url in params_urls.items():
        if needs_file_url(f, ctx, url):
            fetch_http_check_size(name=f, url=url)
    filenames = [*vmfb_filenames, *params_filenames, *mlir_filenames]
    return filenames


if __name__ == "__main__":
    iree_build_main()
