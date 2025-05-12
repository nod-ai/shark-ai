# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import subprocess

from iree.build import *

from shortfin_apps.utils import *

ARTIFACT_VERSION = "04082025"
SDXL_CONFIG_BUCKET = f"https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/{ARTIFACT_VERSION}/configs/"
SUPPORTED_TARGETS = ["gfx942", "gfx1100", "gfx1201"]
SUPPORTED_PRECISIONS = ["i8", "fp8", "fp8_ocp"]


def get_configs(
    model,
    target,
    artifacts_dir,
    model_config: str = None,
    tuning_spec: str = None,
    flagfile: str = None,
):
    # Returns one set of config artifacts.

    artifacts = {
        "model_config": model_config,
        "flagfile": flagfile,
        "tuning_spec": tuning_spec,
    }
    needed = {}
    for a in artifacts.keys():
        if artifacts[a] and not os.path.exists(artifacts[a]):
            needed[a] = artifacts[a]

    if needed:
        downloads = download_configs(artifacts_dir, model)
        complete = None
        for d in downloads:
            for a in needed:
                if str(artifacts[a]) in d:
                    artifacts[a] = d
                    complete = a
            if complete:
                del needed[complete]
                complete = None

        if needed:
            raise FileNotFoundError(str(needed))

    for a in artifacts.keys():
        if artifacts[a] and not os.path.exists(artifacts[a]):
            raise FileNotFoundError(artifacts[a])

    return artifacts


def download_configs(artifacts_dir, model):
    # Downloads artifacts from Azure using the sdxlconfig entrypoint.

    cfg_builder_args = [
        sys.executable,
        "-m",
        "iree.build",
        str(__file__),
        f"--output-dir={artifacts_dir}",
        f"--model={model}",
    ]
    outs = subprocess.check_output(cfg_builder_args).decode()
    outs_paths = outs.splitlines()

    return outs_paths


@entrypoint(description="Retreives a set of SDXL configuration files.")
def sdxlconfig(
    model=cl_arg("model", type=str, default="sdxl", help="Model architecture"),
):
    ctx = executor.BuildContext.current()
    update = needs_update(ctx, ARTIFACT_VERSION)

    model_config_filenames = []
    for p in SUPPORTED_PRECISIONS:
        model_config_filenames.extend(
            [f"sdxl_config_{p}.json", f"sdxl_config_{p}_sched_unet.json"]
        )

    model_config_urls = get_url_map(model_config_filenames, SDXL_CONFIG_BUCKET)
    for f, url in model_config_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)

    flagfile_filenames = []
    tuning_filenames = []
    for tgt in SUPPORTED_TARGETS:
        flagfile_filenames.extend([f"{model}_flagfile_{tgt}.txt"])
        tuning_filenames.extend([f"attention_and_matmul_spec_gfx942.mlir"])

    flagfile_urls = get_url_map(flagfile_filenames, SDXL_CONFIG_BUCKET)
    for f, url in flagfile_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)

    tuning_urls = get_url_map(tuning_filenames, SDXL_CONFIG_BUCKET)
    for f, url in tuning_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)

    filenames = [
        *model_config_filenames,
        *flagfile_filenames,
        *tuning_filenames,
    ]
    return filenames


if __name__ == "__main__":
    iree_build_main()
