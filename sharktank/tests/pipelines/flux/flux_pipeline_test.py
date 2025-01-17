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
    #export_flux_pipeline_iree_parameters,
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
        reference_image_output = reference_model(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=3.5
        ).images[0]
        reference_output = torch.tensor(numpy.array(reference_image_output)).to(dtype=dtype)

        target_output = target_model(
            prompt=prompt,
            height=1024,
            width=1024,
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


@pytest.mark.usefixtures("caching", "get_model_artifacts", "path_prefix")
class FluxPipelineIreeTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        if self.path_prefix is None:
            self.path_prefix = f"{self._temp_dir}/"

    def runTestFluxPipelineIreeCompare(
        self,
        reference_dtype: torch.dtype,
        target_dtype: torch.dtype,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ):
        """Compare IREE pipeline against eager execution."""
        # Initialize reference model
        reference_model = FluxPipeline(
            t5_path="/data/t5-v1_1-xxl/model.gguf",
            clip_path="/data/flux/FLUX.1-dev/text_encoder/model.irpa",
            transformer_path="/data/flux/FLUX.1-dev/transformer/model.irpa",
            ae_path="/data/flux/FLUX.1-dev/vae/model.irpa",
            t5_tokenizer_path="/data/flux/FLUX.1-dev/tokenizer_2/",
            clip_tokenizer_path="/data/flux/FLUX.1-dev/tokenizer/",
            dtype=reference_dtype,
        )

        # Create input tokens
        t5_tokenizer = T5Tokenizer.from_pretrained("/data/flux/FLUX.1-dev/tokenizer_2/")
        clip_tokenizer = CLIPTokenizer.from_pretrained("/data/flux/FLUX.1-dev/tokenizer/")
        
        prompt = "a photo of a forest with mist"
        t5_prompt_ids = torch.tensor([t5_tokenizer(prompt).input_ids], dtype=torch.long)
        clip_prompt_ids = torch.tensor([clip_tokenizer(prompt).input_ids], dtype=torch.long)
        latents = reference_model._get_noise(
            1,
            1024,
            1024,
            seed=12345,
        ).to(dtype=target_dtype) # TODO: it isn't great to be getting this from the reference model
        
        input_args = OrderedDict([
            ("t5_prompt_ids", t5_prompt_ids),
            ("clip_prompt_ids", clip_prompt_ids),
            ("latents", latents)
        ])
        batch_size = t5_prompt_ids.shape[0]

        # Get reference result
        reference_result = reference_model.forward(t5_prompt_ids, clip_prompt_ids, latents)

        # Export and compile for IREE
        target_dtype_name = dtype_to_serialized_short_name(target_dtype)
        target_path_prefix = f"{self.path_prefix}flux_pipeline_{target_dtype_name}"

        parameters_path = f"/data/flux/FLUX.1-dev/"
        # if not self.caching or not os.path.exists(parameters_path):
        #     export_flux_pipeline_iree_parameters(
        #         "/data/flux/FLUX.1-dev",
        #         parameters_path,
        #         dtype=target_dtype,
        #     )

        mlir_path = f"{target_path_prefix}.mlir"
        if not self.caching or not os.path.exists(mlir_path):
            export_flux_pipeline_mlir(
                parameters_path,
                batch_sizes=[batch_size],
                mlir_output_path=mlir_path,
                dtype=target_dtype
            )

        iree_module_path = f"{target_path_prefix}.vmfb"
        if not self.caching or not os.path.exists(iree_module_path):
            iree.compiler.compile_file(
                mlir_path,
                output_file=iree_module_path,
                extra_args=[
                    "--iree-hal-target-device=hip",
                    "--iree-hip-target=gfx942",
                    "--iree-opt-const-eval=false",
                    "--iree-opt-strip-assertions=true",
                    "--iree-global-opt-propagate-transposes=true",
                    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=true",
                    "--iree-dispatch-creation-enable-aggressive-fusion=true",
                    "--iree-opt-aggressively-propagate-transposes=true",
                    "--iree-opt-outer-dim-concat=true",
                    "--iree-vm-target-truncate-unsupported-floats",
                    "--iree-llvmgpu-enable-prefetch=true",
                    "--iree-opt-data-tiling=false",
                    "--iree-codegen-gpu-native-math-precision=true",
                    "--iree-codegen-llvmgpu-use-vector-distribution",
                    "--iree-hip-waves-per-eu=2",
                    "--iree-execution-model=async-external",
                    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
                ],
            )

        # Run with IREE
        iree_devices = get_iree_devices(driver="hip", device_count=1)
        iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
            module_path=iree_module_path,
            devices=iree_devices,
            parameters_path=parameters_path,
        )
        iree_args = prepare_iree_module_function_args(
            args=flatten_for_iree_signature(input_args),
            devices=iree_devices,
        )
        iree_result = iree_to_torch(
            *run_iree_module_function(
                module=iree_module,
                vm_context=iree_vm_context,
                args=iree_args,
                driver="hip",
                function_name=f"forward_bs{batch_size}",
                trace_path_prefix=f"{target_path_prefix}_iree_",
            )
        )
        iree_result = [
            ops.to(iree_result[i], dtype=reference_result[i].dtype)
            for i in range(len(reference_result))
        ]

        torch.testing.assert_close(reference_result, iree_result, atol=atol, rtol=rtol)

    @with_flux_data
    def testFluxPipelineIreeF32(self):
        """Test F32 IREE pipeline against eager execution."""
        self.runTestFluxPipelineIreeCompare(
            reference_dtype=torch.float32,
            target_dtype=torch.float32,
            atol=1e-4,
            rtol=2.0e-3,
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="BF16 vs F32 accuracy needs investigation",
    )
    @with_flux_data
    def testFluxPipelineIreeBF16vsF32(self):
        """Test BF16 IREE pipeline against F16 eager execution."""
        self.runTestFluxPipelineIreeCompare(
            reference_dtype=torch.float32,
            target_dtype=torch.bfloat16,
            atol=1e-2,
            rtol=1.6e-2,
        )
