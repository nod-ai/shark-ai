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

from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import dataclass_json, Undefined

import shortfin.array as sfnp
import json as json

str_to_dtype = {
    "int8": sfnp.int8,
    "float8": sfnp.opaque8,
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

    # Height and Width, respectively, for which Unet and VAE are compiled. e.g. [[512, 512], [1024, 1024]]
    dims: list[list[int]]

    # Scheduler id.
    scheduler_id: str = "EulerDiscrete"

    base_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Batch sizes by submodel. Should be a dict of component string ids as keys and list(int) of their batch sizes.
    batch_sizes: dict = field(default_factory=dict)

    # Module names of exported components.
    module_names: dict = field(default_factory=dict)

    # Function names of exported components.
    function_names: dict = field(default_factory=dict)

    # Classifer free guidance mode. If set to false, only positive prompts will matter.
    cfg_mode = True

    # DTypes (not necessarily weights precision, just I/O of the program):
    clip_dtype: sfnp.DType = sfnp.float16
    unet_dtype: sfnp.DType = sfnp.float16
    vae_dtype: sfnp.DType = sfnp.float16

    # Whether to use amdsharktank punet. This is the only validated path for now.
    use_punet: bool = True

    # Which quantization type. i8, fp8, fp8_ocp. Gets propagated to punet model instantiation.
    # Under the hood, punet always uses mixed precision. the fp8/fp8_ocp variants only change the sdpa dtype.
    # "i8" here means "i8 model with fp16 attention".
    # Pure FP16 is not currently implemented.
    unet_quant_dtype: str = "i8"
    use_scheduled_unet: bool = False

    # ABI of the module.
    module_abi_version: int = 1

    @property
    def all_batch_sizes(self) -> list:
        bs_lists = list(self.batch_sizes.values())
        union = set.union(*[set(list) for list in bs_lists])
        return union

    @staticmethod
    def load_json(path):
        with open(path, "rt") as f:
            json_text = f.read()
        json_obj = json.loads(json_text)
        raw_params = ModelParams.from_json(json_text)
        if isinstance(raw_params.unet_dtype, str):
            raw_params.unet_dtype = str_to_dtype[raw_params.unet_dtype]
        return raw_params

    def __repr__(self):
        return (
            f"     base model : {self.base_model_name}\n"
            f"     unet I/0 dtype : {self.unet_dtype}\n"
            f"     unet quant dtype : {self.unet_quant_dtype}\n"
            f"     use punet : {self.use_punet}\n"
            f"     output size (H,W) : {self.dims}\n"
            f"     max token sequence length : {self.max_seq_len}\n"
            f"     classifier free guidance : {self.cfg_mode}\n"
        )
