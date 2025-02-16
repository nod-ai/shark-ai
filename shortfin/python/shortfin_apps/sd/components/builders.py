# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.build import *
from iree.build.executor import FileNamespace, BuildAction, BuildContext, BuildFile
from iree.turbine.aot.build_actions import turbine_generate
from iree.turbine.aot import (
    ExportOutput,
    FxProgramsBuilder,
    export,
    externalize_module_parameters,
    save_module_parameters,
    decompositions,
)
import itertools
import os
import urllib
import shortfin.array as sfnp
import copy
import re

from shortfin_apps.sd.components.config_struct import ModelParams

this_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(this_dir)
default_config_json = os.path.join(parent, "examples", "sdxl_config_i8.json")

dtype_to_filetag = {
    sfnp.float16: "fp16",
    sfnp.float32: "fp32",
    sfnp.int8: "i8",
    sfnp.bfloat16: "bf16",
}

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


def create_safe_name(hf_model_name, model_name_str=""):
    if not model_name_str:
        model_name_str = ""
    if model_name_str != "" and (not model_name_str.startswith("_")):
        model_name_str = "_" + model_name_str

    safe_name = hf_model_name.split("/")[-1].strip() + model_name_str
    safe_name = re.sub("-", "_", safe_name)
    safe_name = re.sub(r"\.", "_", safe_name)
    return safe_name


def get_params_filenames(model_params: ModelParams, model=None, splat: bool = False):
    params_filenames = []
    base = (
        "stable_diffusion_xl_base_1_0"
        if model_params.base_model_name.lower() in ["sdxl"]
        else create_safe_name(model_params.base_model_name)
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
        else [create_safe_name(model_params.base_model_name)]
    )
    if model_params.use_scheduled_unet:
        denoise_dict = {
            "unet": "scheduled_unet",
        }
    elif model_params.use_i8_punet:
        denoise_dict = {
            "unet": "punet",
            "scheduler": model_params.scheduler_id + "Scheduler",
        }
    else:
        denoise_dict = {
            "unet": "unet",
            "scheduler": model_params.scheduler_id + "Scheduler",
        }
    mod_names = {
        "clip": "clip",
        "vae": "vae",
    }
    mod_names.update(denoise_dict)
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
    try:
        out_file = ctx.allocate_file(filename, namespace=namespace).get_fs_path()
        filekey = os.path.join(ctx.path, filename)
        ctx.executor.all[filekey] = None
    except RuntimeError:
        return False
    needed = True
    if os.path.exists(out_file):
        if url and not is_valid_size(out_file, url):
            return True
        else:
            return False
    return True


def needs_compile(filename, target, ctx):
    vmfb_name = f"{filename}_{target}.vmfb"
    namespace = FileNamespace.BIN
    return needs_file(vmfb_name, ctx, namespace=namespace)


def get_cached_vmfb(filename, target, ctx):
    vmfb_name = f"{filename}_{target}.vmfb"
    return ctx.allocate_file(vmfb_name, namespace=FileNamespace.BIN)


def get_cached(filename, ctx):
    return ctx.allocate_file(filename, namespace=FileNamespace.GEN)


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


def parse_mlir_name(mlir_path):
    terms = mlir_path.split(".mlir")[0].split("_")
    bs_term = [x for x in terms if "bs" in x]
    batch_size = int(bs_term[0].split("bs")[-1])
    dims_match = re.search(r"_(\d+)x(\d+)_", mlir_path)

    if dims_match:
        height = int(dims_match.group(1))
        width = int(dims_match.group(2))
        decomp_attn = False
    else:
        height = None
        width = None
        decomp_attn = True
    precision = [x for x in terms if x in ["i8", "fp8", "fp16", "fp32"]][0]
    max_length = 64
    return batch_size, height, width, decomp_attn, precision, max_length


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
    quant_paths=None,
) -> ExportOutput:
    import torch

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
        if isinstance(external_weights_file, BuildFile):
            external_weights_path = external_weights_file.get_fs_path()
        elif external_weights_file:
            external_weights_path = external_weights_file
        else:
            external_weights_path = None
        if component == "clip":
            from sharktank.torch_exports.sdxl.clip import get_clip_model_and_inputs

            module_name = "compiled_clip"
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
            model.to("cpu")
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
            t_ver = torch.__version__
            if any(key in t_ver for key in ["2.6.", "2.3."]):
                print(
                    "You have a torch version that is unstable for this export and may encounter export or compile-time issues: {t_ver}. The reccommended versions are 2.4.1 - 2.5.1"
                )
            from sharktank.torch_exports.sdxl.unet import (
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
                external_weights_path,
                quant_paths,
            )
            if component == "scheduled_unet":
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

            else:
                return export(
                    model, kwargs=sample_forward_inputs, module_name="compiled_punet"
                )

        elif component == "vae":
            from sharktank.torch_exports.sdxl.vae import get_vae_model_and_inputs

            module_name = "compiled_vae"
            model, encode_args, decode_args = get_vae_model_and_inputs(
                hf_model_name, height, width, precision=precision, batch_size=batch_size
            )
            model.to("cpu")
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
        if external_weights_path:
            save_module_parameters(external_weights_path, model)
    module = export(fxb, module_name=module_name)
    return module


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
    quant_paths=cl_arg(
        "quant-paths", default=None, help="Path for quantized punet model artifacts."
    ),
    force_update=cl_arg("force-update", default=False, help="Force update artifacts."),
):
    model_params = ModelParams.load_json(model_json)
    ctx = executor.BuildContext.current()
    update = needs_update(ctx)

    mlir_bucket = SDXL_BUCKET + "mlir/"
    vmfb_bucket = SDXL_BUCKET + "vmfbs/"
    if "gfx" in target:
        target = "amdgpu-" + target

    params_filenames = get_params_filenames(model_params, model=model, splat=splat)
    mlir_filenames = get_mlir_filenames(model_params, model)
    vmfb_filenames = get_vmfb_filenames(model_params, model=model, target=target)

    if build_preference == "export":
        for mlir_path in mlir_filenames:
            if needs_file(mlir_path, ctx) or force_update in [True, "True"]:
                (
                    batch_size,
                    height,
                    width,
                    decomp_attn,
                    precision,
                    max_length,
                ) = parse_mlir_name(mlir_path)
                mod = turbine_generate(
                    export_sdxl_model,
                    hf_model_name=model_params.base_model_name,
                    component=model,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    precision=precision,
                    max_length=max_length,
                    external_weights="irpa",
                    external_weights_file=params_filenames[
                        0
                    ],  # Should only get one per invocation
                    decomp_attn=decomp_attn,
                    name=mlir_path.split(".mlir")[0],
                    out_of_process=False,
                )
            else:
                get_cached(mlir_path, ctx)
        for params_path in params_filenames:
            get_cached(params_path, ctx)

    else:
        mlir_urls = get_url_map(mlir_filenames, mlir_bucket)
        for f, url in mlir_urls.items():
            if update or needs_file(f, ctx, url):
                fetch_http(name=f, url=url)
            else:
                get_cached(f, ctx)
        params_urls = get_url_map(params_filenames, SDXL_WEIGHTS_BUCKET)
        for f, url in params_urls.items():
            if needs_file(f, ctx, url):
                fetch_http_check_size(name=f, url=url)
            else:
                get_cached(f, ctx)
    if build_preference != "precompiled":
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
        vmfb_urls = get_url_map(vmfb_filenames, vmfb_bucket)
        for f, url in vmfb_urls.items():
            if update or needs_file(f, ctx, url):
                fetch_http(name=f, url=url)

    filenames = [*vmfb_filenames, *params_filenames, *mlir_filenames]
    return filenames


if __name__ == "__main__":
    iree_build_main()
