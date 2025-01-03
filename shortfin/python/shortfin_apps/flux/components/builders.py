# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.build import *
from iree.build.executor import FileNamespace, BuildAction, BuildContext, BuildFile
import itertools
import os
import urllib
import shortfin.array as sfnp
import copy

from shortfin_apps.flux.components.config_struct import ModelParams

this_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(this_dir)
default_config_json = os.path.join(parent, "examples", "flux_dev_config_mixed.json")

dtype_to_filetag = {
    "bfloat16": "bf16",
    "float32": "fp32",
    "float16": "fp16",
    sfnp.float32: "fp32",
    sfnp.bfloat16: "bf16",
}

ARTIFACT_VERSION = "12032024"
SDXL_BUCKET = (
    f"https://sharkpublic.blob.core.windows.net/sharkpublic/flux.1/{ARTIFACT_VERSION}/"
)
SDXL_WEIGHTS_BUCKET = (
    "https://sharkpublic.blob.core.windows.net/sharkpublic/flux.1/weights/"
)


def filter_by_model(filenames, model):
    if not model:
        return filenames
    filtered = []
    for i in filenames:
        if model == "t5xxl" and i == "google__t5_v1_1_xxl_encoder_fp32.irpa":
            filtered.extend([i])
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
    base = "flux_dev" if not model_params.is_schnell else model_params.base_model_name
    modnames = ["clip", "sampler", "vae"]
    mod_precs = [
        dtype_to_filetag[model_params.clip_dtype],
        dtype_to_filetag[model_params.sampler_dtype],
        dtype_to_filetag[model_params.vae_dtype],
    ]
    if splat == "True":
        for idx, mod in enumerate(modnames):
            params_filenames.extend(
                ["_".join([mod, "splat", f"{mod_precs[idx]}.irpa"])]
            )
    else:
        for idx, mod in enumerate(modnames):
            params_filenames.extend([base + "_" + mod + "_" + mod_precs[idx] + ".irpa"])

    # this is a hack
    params_filenames.extend(["google__t5_v1_1_xxl_encoder_fp32.irpa"])

    return filter_by_model(params_filenames, model)


def get_file_stems(model_params: ModelParams):
    file_stems = []
    base = ["flux_dev" if not model_params.is_schnell else model_params.base_model_name]
    mod_names = {
        "clip": "clip",
        "t5xxl": "t5xxl",
        "sampler": "sampler",
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
        if mod in ["sampler"]:
            ord_params.extend([[str(model_params.max_seq_len)]])
        elif mod == "clip":
            ord_params.extend([[str(model_params.clip_max_seq_len)]])
        if mod in ["sampler", "vae"]:
            dims = []
            for dim_pair in model_params.dims:
                dim_pair_str = [str(d) for d in dim_pair]
                dims.extend(["x".join(dim_pair_str)])
            ord_params.extend([dims])

        dtype_str = dtype_to_filetag[
            getattr(model_params, f"{mod}_dtype", sfnp.float32)
        ]
        ord_params.extend([[dtype_str]])
        for x in list(itertools.product(*ord_params)):
            file_stems.extend(["_".join(x)])
    return file_stems


def get_url_map(filenames: list[str], bucket: str):
    file_map = {}
    for filename in filenames:
        file_map[filename] = f"{bucket}{filename}"
    return file_map


def needs_update(ctx):
    stamp = ctx.allocate_file("version.txt")
    stamp_path = stamp.get_fs_path()
    if os.path.exists(stamp_path):
        with open(stamp_path, "r") as s:
            ver = s.read()
        if ver != ARTIFACT_VERSION:
            return True
    else:
        with open(stamp_path, "w") as s:
            s.write(ARTIFACT_VERSION)
        return True
    return False


def needs_file(filename, ctx, url=None, namespace=FileNamespace.GEN):
    out_file = ctx.allocate_file(filename, namespace=namespace).get_fs_path()
    needed = True
    if os.path.exists(out_file):
        if url:
            needed = False
            # needed = not is_valid_size(out_file, url)
        if not needed:
            return False
    filekey = os.path.join(ctx.path, filename)
    ctx.executor.all[filekey] = None
    return True


def needs_compile(filename, target, ctx):
    vmfb_name = f"{filename}_{target}.vmfb"
    namespace = FileNamespace.BIN
    return needs_file(vmfb_name, ctx, namespace=namespace)


def get_cached_vmfb(filename, target, ctx):
    vmfb_name = f"{filename}_{target}.vmfb"
    return ctx.file(vmfb_name)


def is_valid_size(file_path, url):
    if not url:
        return True
    with urllib.request.urlopen(url) as response:
        content_length = response.getheader("Content-Length")
    local_size = get_file_size(str(file_path))
    if content_length:
        content_length = int(content_length)
        if content_length != local_size:
            return False
    return True


def get_file_size(file_path):
    """Gets the size of a local file in bytes as an integer."""

    file_stats = os.stat(file_path)
    return file_stats.st_size


def fetch_http_check_size(*, name: str, url: str) -> BuildFile:
    context = BuildContext.current()
    output_file = context.allocate_file(name)
    action = FetchHttpWithCheckAction(
        url=url, output_file=output_file, desc=f"Fetch {url}", executor=context.executor
    )
    output_file.deps.add(action)
    return output_file


class FetchHttpWithCheckAction(BuildAction):
    def __init__(self, url: str, output_file: BuildFile, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.output_file = output_file

    def _invoke(self, retries=4):
        path = self.output_file.get_fs_path()
        self.executor.write_status(f"Fetching URL: {self.url} -> {path}")
        try:
            urllib.request.urlretrieve(self.url, str(path))
        except urllib.error.HTTPError as e:
            if retries > 0:
                retries -= 1
                self._invoke(retries=retries)
            else:
                raise IOError(f"Failed to fetch URL '{self.url}': {e}") from None
        local_size = get_file_size(str(path))
        try:
            with urllib.request.urlopen(self.url) as response:
                content_length = response.getheader("Content-Length")
            if content_length:
                content_length = int(content_length)
                if content_length != local_size:
                    raise IOError(
                        f"Size of downloaded artifact does not match content-length header! {content_length} != {local_size}"
                    )
        except IOError:
            if retries > 0:
                retries -= 1
                self._invoke(retries=retries)


@entrypoint(description="Retreives a set of SDXL submodels.")
def flux(
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
    update = needs_update(ctx)

    mlir_bucket = SDXL_BUCKET + "mlir/"
    vmfb_bucket = SDXL_BUCKET + "vmfbs/"
    if "gfx" in target:
        target = "amdgpu-" + target

    mlir_filenames = get_mlir_filenames(model_params, model)
    mlir_urls = get_url_map(mlir_filenames, mlir_bucket)
    for f, url in mlir_urls.items():
        if update or needs_file(f, ctx, url):
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
            if update or needs_file(f, ctx, url):
                fetch_http(name=f, url=url)

    params_filenames = get_params_filenames(model_params, model=model, splat=splat)
    params_urls = get_url_map(params_filenames, SDXL_WEIGHTS_BUCKET)
    for f, url in params_urls.items():
        if needs_file(f, ctx, url):
            fetch_http_check_size(name=f, url=url)
    filenames = [*vmfb_filenames, *params_filenames, *mlir_filenames]
    return filenames


if __name__ == "__main__":
    iree_build_main()
