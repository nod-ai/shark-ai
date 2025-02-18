# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.build import *

from shortfin_apps.utils import *

ARTIFACT_VERSION = "12032024"
FLUX_CONFIG_BUCKET = f"https://sharkpublic.blob.core.windows.net/sharkpublic/flux.1/{ARTIFACT_VERSION}/configs/"


@entrypoint(description="Retreives a set of FLUX configuration files.")
def fluxconfig(
    target=cl_arg(
        "target",
        default="gfx942",
        help="IREE target architecture.",
    ),
    model=cl_arg("model", type=str, default="flux", help="Model architecture"),
    topology=cl_arg(
        "topology",
        type=str,
        default=None,
        help="System topology configfile keyword",
    ),
):
    ctx = executor.BuildContext.current()
    update = needs_update(ctx, ARTIFACT_VERSION)

    if topology:
        topology_config_filenames = [f"topology_config_{topology}.txt"]
        topology_config_urls = get_url_map(
            topology_config_filenames, FLUX_CONFIG_BUCKET
        )
        for f, url in topology_config_urls.items():
            if update or needs_file(f, ctx):
                fetch_http(name=f, url=url)

    tuning_filenames = (
        [f"attention_and_matmul_spec_{target}.mlir"] if target == "gfx942" else []
    )
    tuning_urls = get_url_map(tuning_filenames, FLUX_CONFIG_BUCKET)
    for f, url in tuning_urls.items():
        if update or needs_file(f, ctx):
            fetch_http(name=f, url=url)
    filenames = [
        *tuning_filenames,
    ]
    if topology:
        filenames.extend(
            [
                *topology_config_filenames,
            ]
        )
    return filenames


if __name__ == "__main__":
    iree_build_main()
