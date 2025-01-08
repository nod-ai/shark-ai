# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import sys

import torch

from iree.turbine import aot

from sharktank.types import Dataset
from sharktank.models.vae.model import VaeDecoderModel
from sharktank.models.vae.tools.diffuser_ref import run_torch_vae
from sharktank.models.vae.tools.run_vae import export_vae
from sharktank.models.vae.tools.sample_data import get_random_inputs

from sharktank.models.punet.tools.sample_data import load_inputs, save_outputs
from sharktank.tools.import_hf_dataset import import_hf_dataset
from iree.turbine.aot import FxProgramsBuilder, export, decompositions
from sharktank.utils.hf_datasets import get_dataset
import unittest
import pytest
from huggingface_hub import hf_hub_download
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
import iree.compiler
from collections import OrderedDict

with_vae_data = pytest.mark.skipif("not config.getoption('with_vae_data')")


@with_vae_data
class VaeSDXLDecoderTest(unittest.TestCase):
    def setUp(self):
        hf_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        hf_hub_download(
            repo_id=hf_model_id,
            local_dir="sdxl_vae",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/config.json",
        )
        hf_hub_download(
            repo_id=hf_model_id,
            local_dir="sdxl_vae",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/diffusion_pytorch_model.safetensors",
        )
        hf_hub_download(
            repo_id="amd-shark/sdxl-quant-models",
            local_dir="sdxl_vae",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/vae.safetensors",
        )
        torch.manual_seed(12345)
        f32_dataset = import_hf_dataset(
            "sdxl_vae/vae/config.json",
            ["sdxl_vae/vae/diffusion_pytorch_model.safetensors"],
        )
        f32_dataset.save("sdxl_vae/vae_f32.irpa", io_report_callback=print)
        f16_dataset = import_hf_dataset(
            "sdxl_vae/vae/config.json", ["sdxl_vae/vae/vae.safetensors"]
        )
        f16_dataset.save("sdxl_vae/vae_f16.irpa", io_report_callback=print)

    def testCompareF32EagerVsHuggingface(self):
        dtype = getattr(torch, "float32")
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1)
        ref_results = run_torch_vae("sdxl_vae", inputs)

        ds = Dataset.load("sdxl_vae/vae_f32.irpa", file_type="irpa")
        model = VaeDecoderModel.from_dataset(ds).to(device="cpu")

        results = model.forward(inputs)

        torch.testing.assert_close(ref_results, results)

    @pytest.mark.skip(reason="running fp16 on cpu is extremely slow")
    def testCompareF16EagerVsHuggingface(self):
        dtype = getattr(torch, "float32")
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1)
        ref_results = run_torch_vae("sdxl_vae", inputs)

        ds = Dataset.load("sdxl_vae/vae_f16.irpa", file_type="irpa")
        model = VaeDecoderModel.from_dataset(ds).to(device="cpu")

        results = model.forward(inputs.to(torch.float16))

        torch.testing.assert_close(ref_results, results)

    @pytest.mark.xfail(
        reason="Waiting on fix for https://github.com/iree-org/iree/issues/19623"
    )
    def testVaeIreeVsHuggingFace(self):
        dtype = getattr(torch, "float32")
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1)
        ref_results = run_torch_vae("sdxl_vae", inputs)

        ds_f16 = Dataset.load("sdxl_vae/vae_f16.irpa", file_type="irpa")
        ds_f32 = Dataset.load("sdxl_vae/vae_f32.irpa", file_type="irpa")

        model_f16 = VaeDecoderModel.from_dataset(ds_f16).to(device="cpu")
        model_f32 = VaeDecoderModel.from_dataset(ds_f32).to(device="cpu")

        # TODO: Decomposing attention due to https://github.com/iree-org/iree/issues/19286, remove once issue is resolved
        module_f16 = export_vae(model_f16, inputs.to(torch.float16), True)
        module_f32 = export_vae(model_f32, inputs, True)

        module_f16.save_mlir("sdxl_vae/vae_f16.mlir")
        module_f32.save_mlir("sdxl_vae/vae_f32.mlir")
        extra_args = [
            "--iree-hal-target-backends=rocm",
            "--iree-hip-target=gfx942",
            "--iree-opt-const-eval=false",
            "--iree-opt-strip-assertions=true",
            "--iree-global-opt-propagate-transposes=true",
            "--iree-opt-outer-dim-concat=true",
            "--iree-llvmgpu-enable-prefetch=true",
            "--iree-hip-waves-per-eu=2",
            "--iree-dispatch-creation-enable-aggressive-fusion=true",
            "--iree-codegen-llvmgpu-use-vector-distribution=true",
            "--iree-execution-model=async-external",
            "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
        ]

        iree.compiler.compile_file(
            "sdxl_vae/vae_f16.mlir",
            output_file="sdxl_vae/vae_f16.vmfb",
            extra_args=extra_args,
        )
        iree.compiler.compile_file(
            "sdxl_vae/vae_f32.mlir",
            output_file="sdxl_vae/vae_f32.vmfb",
            extra_args=extra_args,
        )

        iree_devices = get_iree_devices(driver="hip", device_count=1)

        iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
            module_path="sdxl_vae/vae_f16.vmfb",
            devices=iree_devices,
            parameters_path="sdxl_vae/vae_f16.irpa",
        )

        input_args = OrderedDict([("inputs", inputs.to(torch.float16))])
        iree_args = flatten_for_iree_signature(input_args)

        iree_args = prepare_iree_module_function_args(
            args=iree_args, devices=iree_devices
        )
        iree_result = run_iree_module_function(
            module=iree_module,
            vm_context=iree_vm_context,
            args=iree_args,
            driver="hip",
            function_name="forward",
        )[0].to_host()
        # TODO: Verify these numerics are good or if tolerances are too loose
        # TODO: Upload IR on passing tests to keep https://github.com/iree-org/iree/blob/main/experimental/regression_suite/shark-test-suite-models/sdxl/test_vae.py at latest
        torch.testing.assert_close(
            ref_results.to(torch.float16),
            torch.from_numpy(iree_result),
            atol=5e-2,
            rtol=4e-1,
        )

        iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
            module_path="sdxl_vae/vae_f32.vmfb",
            devices=iree_devices,
            parameters_path="sdxl_vae/vae_f32.irpa",
        )

        input_args = OrderedDict([("inputs", inputs)])
        iree_args = flatten_for_iree_signature(input_args)

        iree_args = prepare_iree_module_function_args(
            args=iree_args, devices=iree_devices
        )
        iree_result = run_iree_module_function(
            module=iree_module,
            vm_context=iree_vm_context,
            args=iree_args,
            driver="hip",
            function_name="forward",
        )[0].to_host()
        # TODO: Upload IR on passing tests
        torch.testing.assert_close(
            ref_results, torch.from_numpy(iree_result), atol=3e-5, rtol=6e-6
        )


if __name__ == "__main__":
    unittest.main()
