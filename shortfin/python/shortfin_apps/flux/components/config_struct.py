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
    "bfloat16": sfnp.bfloat16,
    "float32": sfnp.float32,
}


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelParams:
    """Parameters for a specific set of compiled SD submodels, sufficient to do batching /
    invocations."""

    # Batch sizes that each stage is compiled for. These are expected to be
    # functions exported from the model with suffixes of "_bs{batch_size}". Must
    # be in ascending order.
    clip_batch_sizes: list[int]

    t5xxl_batch_sizes: list[int]

    sampler_batch_sizes: list[int]

    vae_batch_sizes: list[int]

    # Height and Width, respectively, for which sampler and VAE are compiled. e.g. [[512, 512], [1024, 1024]]
    dims: list[list[int]]

    base_model_name: str = "flux_dev"
    clip_max_seq_len: int = 77
    clip_module_name: str = "compiled_flux_text_encoder"
    clip_fn_name: str = "encode_prompts"
    clip_dtype: sfnp.DType = sfnp.bfloat16

    max_seq_len: int = 512
    t5xxl_module_name: str = "module"
    t5xxl_fn_name: str = "forward_bs4"
    t5xxl_dtype: sfnp.DType = sfnp.bfloat16

    # Channel dim of latents.
    num_latents_channels: int = 16

    sampler_module_name: str = ""
    sampler_fn_name: str = "main_graph"
    sampler_dtype: sfnp.DType = sfnp.float32

    vae_module_name: str = "compiled_vae"
    vae_fn_name: str = "decode"
    vae_dtype: sfnp.DType = sfnp.float32

    # Whether model is "schnell" (fast) or not. This is roughly equivalent to "turbo" from SDXL.
    # It cuts batch dims in half for sampling/encoding and removes negative prompt functionality.
    is_schnell: bool = False

    # ABI of the module.
    module_abi_version: int = 1

    @property
    def max_clip_batch_size(self) -> int:
        return self.clip_batch_sizes[-1]

    @property
    def max_sampler_batch_size(self) -> int:
        return self.sampler_batch_sizes[-1]

    @property
    def max_vae_batch_size(self) -> int:
        return self.vae_batch_sizes[-1]

    @property
    def all_batch_sizes(self) -> list:
        return [self.clip_batch_sizes, self.sampler_batch_sizes, self.vae_batch_sizes]

    @property
    def max_batch_size(self):
        return max(self.all_batch_sizes)

    @staticmethod
    def load_json(path: Path | str):
        with open(path, "rt") as f:
            json_text = f.read()
        raw_params = ModelParams.from_json(json_text)
        for i in ["sampler_dtype", "t5xxl_dtype", "clip_dtype", "vae_dtype"]:
            if isinstance(i, str):
                setattr(raw_params, i, str_to_dtype[getattr(raw_params, i)])
        return raw_params
