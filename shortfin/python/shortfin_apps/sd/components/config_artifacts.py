# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.build import *
from iree.build.executor import FileNamespace
import os

ARTIFACT_VERSION = "01062025"
SDXL_CONFIG_BUCKET = f"https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/{ARTIFACT_VERSION}/configs/"


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


def needs_file(filename, ctx, namespace=FileNamespace.GEN):
    out_file = ctx.allocate_file(filename, namespace=namespace).get_fs_path()
    if os.path.exists(out_file):
        needed = False
    else:
        # name_path = "bin" if namespace == FileNamespace.BIN else ""
        # if name_path:
        #     filename = os.path.join(name_path, filename)
        filekey = os.path.join(ctx.path, filename)
        ctx.executor.all[filekey] = None
        needed = True
    return needed


@entrypoint(description="Retreives a set of SDXL configuration files.")
def sdxlconfig(
    target=cl_arg(
        "target",
        default="gfx942",
        help="IREE target architecture.",
    ),
    model=cl_arg("model", type=str, default="sdxl", help="Model architecture"),
    topology=cl_arg(
        "topology",
        type=str,
        default="spx_single",
        help="System topology configfile keyword",
    ),
):
    ctx = executor.BuildContext.current()
    update = needs_update(ctx)

    model_config_filenames = [f"{model}_config_i8.json"]
    model_config_urls = get_url_map(model_config_filenames, SDXL_CONFIG_BUCKET)
    for f, url in model_config_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)

    topology_config_filenames = [f"topology_config_{topology}.txt"]
    topology_config_urls = get_url_map(topology_config_filenames, SDXL_CONFIG_BUCKET)
    for f, url in topology_config_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)

    flagfile_filenames = [f"{model}_flagfile_{target}.txt"]
    flagfile_urls = get_url_map(flagfile_filenames, SDXL_CONFIG_BUCKET)
    for f, url in flagfile_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)

    tuning_filenames = (
        [f"attention_and_matmul_spec_{target}.mlir"] if target == "gfx942" else []
    )
    tuning_urls = get_url_map(tuning_filenames, SDXL_CONFIG_BUCKET)
    for f, url in tuning_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)
    filenames = [
        *model_config_filenames,
        *topology_config_filenames,
        *flagfile_filenames,
        *tuning_filenames,
    ]
    return filenames


if __name__ == "__main__":
    iree_build_main()
