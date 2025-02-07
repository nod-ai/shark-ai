# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# DO NOT SUBMIT: REVIEW AND TEST FILE

"""Tests for Flux text-to-image pipeline."""

import functools
from typing import Optional
import os
from collections import OrderedDict
import pytest
import torch
from unittest import TestCase
import numpy

from transformers import CLIPTokenizer, T5Tokenizer
from diffusers import FluxPipeline as ReferenceFluxPipeline

from sharktank.types import Dataset, dtype_to_serialized_short_name
from sharktank.pipelines.flux import (
    FluxPipeline,
    export_flux_pipeline_mlir,
    export_flux_pipeline_iree_parameters,
)
from sharktank.utils.testing import TempDirTestBase
from sharktank.transforms.dataset import set_float_dtype
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
from sharktank import ops
import iree.compiler

with_flux_data = pytest.mark.skipif("not config.getoption('with_flux_data')")

@pytest.mark.usefixtures("get_model_artifacts")
class FluxPipelineEagerTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.random.manual_seed(12345)
        torch.no_grad()

    @with_flux_data
    def testFluxPipelineAgainstGolden(self):
        """Test against golden outputs from the original Flux pipeline."""
        model = FluxPipeline(
            t5_path="/data/t5-v1_1-xxl/model.gguf",
            clip_path="/data/flux/FLUX.1-dev/text_encoder/model.irpa",
            transformer_path="/data/flux/FLUX.1-dev/transformer/model.irpa",
            ae_path="/data/flux/FLUX.1-dev/vae/model.irpa",
            dtype=torch.bfloat16,
        )

        # Load reference inputs
        with open("/data/flux/test_data/t5_prompt_ids.pt", "rb") as f:
            t5_prompt_ids = torch.load(f)
        with open("/data/flux/test_data/clip_prompt_ids.pt", "rb") as f:
            clip_prompt_ids = torch.load(f)

        # Generate output using forward method directly
        latents = model._get_noise(
            1,
            1024,
            1024,
            seed=12345,
        )
        output = model.forward(
            t5_prompt_ids,
            clip_prompt_ids,
            latents=latents,
            num_inference_steps=1,
            seed=12345,
        )

        # Compare against golden output
        with open("/data/flux/test_data/flux_1_step_output.pt", "rb") as f:
            reference_output = torch.load(f)

        torch.testing.assert_close(output, reference_output) # TODO: why is this not passing?

    def runTestFluxPipelineAgainstReference(
        self,
        dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        """Compare pipeline outputs between different dtypes."""
        # Initialize reference model
        reference_model = ReferenceFluxPipeline.from_pretrained("/data/flux/FLUX.1-dev/")

        # Initialize target model
        target_model = FluxPipeline(
            t5_path="/data/t5-v1_1-xxl/model.gguf",
            clip_path="/data/flux/FLUX.1-dev/text_encoder/model.irpa",
            transformer_path="/data/flux/FLUX.1-dev/transformer/model.irpa",
            ae_path="/data/flux/FLUX.1-dev/vae/model.irpa",
            t5_tokenizer_path="/data/flux/FLUX.1-dev/tokenizer_2/",
            clip_tokenizer_path="/data/flux/FLUX.1-dev/tokenizer/",
            dtype=dtype,
        )

        # Generate outputs using string prompt
        prompt = "a photo of a forest with mist"
        latents = reference_model._get_noise(
            1,
            1024,
            1024,
            seed=12345,
        ).to(dtype=dtype)
        reference_image_output = reference_model(
            prompt=prompt,
            height=1024,
            width=1024,
            latents=latents,
            num_inference_steps=1,
            guidance_scale=3.5
        ).images[0]
        reference_output = torch.tensor(numpy.array(reference_image_output)).to(dtype=dtype)

        target_output = target_model(
            prompt=prompt,
            height=1024,
            width=1024,
            latents=latents,
            num_inference_steps=1,
            guidance_scale=3.5
        )

        torch.testing.assert_close(reference_output, target_output, atol=atol, rtol=rtol)

    @with_flux_data
    def testFluxPipelineF32(self):
        """Test F32 pipeline against reference."""
        self.runTestFluxPipelineAgainstReference(
            dtype=torch.float32,
        )

    @with_flux_data
    def testFluxPipelineBF16(self):
        """Test BF16 pipeline against refence."""
        self.runTestFluxPipelineAgainstReference(
            dtype=torch.bfloat16,
        )


