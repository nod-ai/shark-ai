# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration objects.

Parameters that are intrinsic to a specific model.

Typically represented in something like a Huggingface config.json,
we extend the configuration to enumerate inference boundaries of some given set of compiled modules.
"""

from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json, Undefined

import shortfin.array as sfnp

str_to_dtype = {
    "int8": sfnp.int8,
    "float16": sfnp.float16,
}


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelParams:
    """Parameters for a specific set of compiled SD submodels, sufficient to do batching /
    invocations."""

    # Maximum length of prompt sequence.
    max_seq_len: int

    # Channel dim of latents.
    num_latents_channels: int

    # Batch sizes that each stage is compiled for. These are expected to be
    # functions exported from the model with suffixes of "_bs{batch_size}". Must
    # be in ascending order.
    clip_batch_sizes: list[int]

    # Similarly, batch sizes that the decode stage is compiled for.
    unet_batch_sizes: list[int]

    # Same for VAE.
    vae_batch_sizes: list[int]

    # Same for scheduler.
    scheduler_batch_sizes: list[int]

    # Height and Width, respectively, for which Unet and VAE are compiled. e.g. [[512, 512], [1024, 1024]]
    dims: list[list[int]]

    # Scheduler id.
    scheduler_id: str = "EulerDiscrete"

    base_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # Name of the IREE module for each submodel.
    clip_module_name: str = "compiled_clip"
    unet_module_name: str = "compiled_unet"
    vae_module_name: str = "compiled_vae"
    scheduler_module_name: str = "compiled_scheduler"

    # some unet vmfbs have "main" as entrypoint.
    unet_fn_name: str = "run_forward"

    # Classifer free guidance mode. If set to false, only positive prompts will matter.
    cfg_mode = True

    # DTypes (not necessarily weights precision):
    clip_dtype: sfnp.DType = sfnp.float16
    unet_dtype: sfnp.DType = sfnp.float16
    vae_dtype: sfnp.DType = sfnp.float16

    use_i8_punet: bool = False
    use_scheduled_unet: bool = False

    # ABI of the module.
    module_abi_version: int = 1

    @property
    def max_clip_batch_size(self) -> int:
        return self.clip_batch_sizes[-1]

    @property
    def max_unet_batch_size(self) -> int:
        return self.unet_batch_sizes[-1]

    @property
    def max_vae_batch_size(self) -> int:
        return self.vae_batch_sizes[-1]

    @property
    def all_batch_sizes(self) -> list:
        intersection = list(
            set(self.clip_batch_sizes)
            & set(self.unet_batch_sizes)
            & set(self.vae_batch_sizes)
        )
        return intersection

    @property
    def max_batch_size(self):
        return max(self.all_batch_sizes)

    @staticmethod
    def load_json(path: Path | str):
        with open(path, "rt") as f:
            json_text = f.read()
        raw_params = ModelParams.from_json(json_text)
        if isinstance(raw_params.unet_dtype, str):
            raw_params.unet_dtype = str_to_dtype[raw_params.unet_dtype]
        return raw_params

    def __repr__(self):
        return (
            f"     base model : {self.base_model_name} \n"
            f"     output size (H,W) : {self.dims} \n"
            f"     max token sequence length : {self.max_seq_len} \n"
            f"     classifier free guidance : {self.cfg_mode} \n"
        )
