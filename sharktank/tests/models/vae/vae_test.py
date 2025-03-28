# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict
import logging
import torch
import unittest
import pytest
import iree.runtime
import iree.compiler
from huggingface_hub import hf_hub_download
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import functools
from parameterized import parameterized
import platform


from sharktank.types import Dataset
from sharktank.models.vae.model import VaeDecoderModel
from sharktank.models.vae.tools.diffuser_ref import (
    run_torch_vae,
    convert_vae_decoder_to_hugging_face,
)
from sharktank.models.vae.tools.run_vae import export_vae
from sharktank.models.vae.tools.sample_data import get_random_inputs
from sharktank.tools.import_hf_dataset import import_hf_dataset
from sharktank.utils.iree import (
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    flatten_for_iree_signature,
    device_array_to_host,
)
from sharktank.utils.testing import (
    TempDirTestBase,
    get_iree_compiler_flags,
    is_cpu_condition,
)
from sharktank.models.vae.testing import (
    get_toy_vae_decoder_config,
    make_vae_decoder_random_theta,
)
from sharktank.transforms.dataset import set_float_dtype

logger = logging.getLogger(__name__)

with_vae_data = pytest.mark.skipif("not config.getoption('with_vae_data')")


@with_vae_data
@pytest.mark.usefixtures("get_iree_flags")
class VaeSDXLDecoderTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        hf_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        hf_hub_download(
            repo_id=hf_model_id,
            local_dir=f"{self._temp_dir}",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/config.json",
        )
        hf_hub_download(
            repo_id=hf_model_id,
            local_dir=f"{self._temp_dir}",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/diffusion_pytorch_model.safetensors",
        )
        hf_hub_download(
            repo_id="amd-shark/sdxl-quant-models",
            local_dir=f"{self._temp_dir}",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/vae.safetensors",
        )
        torch.manual_seed(12345)
        f32_dataset = import_hf_dataset(
            f"{self._temp_dir}/vae/config.json",
            [f"{self._temp_dir}/vae/diffusion_pytorch_model.safetensors"],
        )
        f32_dataset.save(
            f"{self._temp_dir}/vae_f32.irpa", io_report_callback=logger.debug
        )
        f16_dataset = import_hf_dataset(
            f"{self._temp_dir}/vae/config.json",
            [f"{self._temp_dir}/vae/vae.safetensors"],
        )
        f16_dataset.save(
            f"{self._temp_dir}/vae_f16.irpa", io_report_callback=logger.debug
        )

    @pytest.mark.expensive
    def testCompareF32EagerVsHuggingface(self):
        dtype = getattr(torch, "float32")
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1)
        ref_results = run_torch_vae(f"{self._temp_dir}", inputs)

        ds = Dataset.load(f"{self._temp_dir}/vae_f32.irpa", file_type="irpa")
        model = VaeDecoderModel.from_dataset(ds).to(device="cpu")

        results = model.forward(inputs)

        torch.testing.assert_close(ref_results, results)

    @pytest.mark.expensive
    @pytest.mark.skip(reason="running fp16 on cpu is extremely slow")
    def testCompareF16EagerVsHuggingface(self):
        dtype = getattr(torch, "float32")
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1)
        ref_results = run_torch_vae(f"{self._temp_dir}", inputs)

        ds = Dataset.load(f"{self._temp_dir}/vae_f16.irpa", file_type="irpa")
        model = VaeDecoderModel.from_dataset(ds).to(device="cpu")

        results = model.forward(inputs.to(torch.float16))

        torch.testing.assert_close(ref_results, results)

    @pytest.mark.expensive
    def testVaeIreeVsHuggingFace(self):
        dtype = getattr(torch, "float32")
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1)
        ref_results = run_torch_vae(f"{self._temp_dir}", inputs)

        ds_f16 = Dataset.load(f"{self._temp_dir}/vae_f16.irpa", file_type="irpa")
        ds_f32 = Dataset.load(f"{self._temp_dir}/vae_f32.irpa", file_type="irpa")

        model_f16 = VaeDecoderModel.from_dataset(ds_f16).to(device="cpu")
        model_f32 = VaeDecoderModel.from_dataset(ds_f32).to(device="cpu")

        # TODO: Decomposing attention due to https://github.com/iree-org/iree/issues/19286, remove once issue is resolved
        module_f16 = export_vae(model_f16, inputs.to(torch.float16), True)
        module_f32 = export_vae(model_f32, inputs, True)

        module_f16.save_mlir(f"{self._temp_dir}/vae_f16.mlir")
        module_f32.save_mlir(f"{self._temp_dir}/vae_f32.mlir")
        extra_args = [
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
        ] + get_iree_compiler_flags(self)

        iree.compiler.compile_file(
            f"{self._temp_dir}/vae_f16.mlir",
            output_file=f"{self._temp_dir}/vae_f16.vmfb",
            extra_args=extra_args,
        )
        iree.compiler.compile_file(
            f"{self._temp_dir}/vae_f32.mlir",
            output_file=f"{self._temp_dir}/vae_f32.vmfb",
            extra_args=extra_args,
        )

        iree_devices = get_iree_devices(driver=self.iree_device, device_count=1)

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=f"{self._temp_dir}/vae_f16.vmfb",
                devices=iree_devices,
                parameters_path=f"{self._temp_dir}/vae_f16.irpa",
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
                device=iree_devices[0],
                function_name="decode",
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
                module_path=f"{self._temp_dir}/vae_f32.vmfb",
                devices=iree_devices,
                parameters_path=f"{self._temp_dir}/vae_f32.irpa",
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
                device=iree_devices[0],
                function_name="decode",
            )[0].to_host()
            return torch.from_numpy(iree_result).clone()

        iree_result = with_iree_device_context(run_iree_module, iree_devices)

        # TODO: Upload IR on passing tests
        torch.testing.assert_close(ref_results, iree_result, atol=3e-5, rtol=6e-6)


@pytest.mark.usefixtures("get_iree_flags")
class VaeFluxDecoderTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)
        self.hf_model_id = "black-forest-labs/FLUX.1-dev"
        self.extra_args = [
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
        ] + get_iree_compiler_flags(self)

    @pytest.mark.expensive
    @with_vae_data
    def testCompareBF16EagerVsHuggingface(self):
        self.download_from_hf()
        self.import_bf16_dataset()

        dtype = torch.bfloat16
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1, config="flux")
        ref_results = run_torch_vae(
            "black-forest-labs/FLUX.1-dev", inputs, 1024, 1024, True, dtype
        )

        ds = Dataset.load(f"{self._temp_dir}/flux_vae_bf16.irpa", file_type="irpa")
        model = VaeDecoderModel.from_dataset(ds).to(device="cpu")

        results = model.forward(inputs)
        # TODO: verify numerics
        torch.testing.assert_close(ref_results, results, atol=3e-2, rtol=3e5)

    @pytest.mark.expensive
    @with_vae_data
    def testCompareF32EagerVsHuggingface(self):
        self.download_from_hf()
        self.import_f32_dataset()

        dtype = torch.float32
        inputs = get_random_inputs(dtype=dtype, device="cpu", bs=1, config="flux")
        ref_results = run_torch_vae(
            "black-forest-labs/FLUX.1-dev", inputs, 1024, 1024, True, dtype
        )

        ds = Dataset.load(f"{self._temp_dir}/flux_vae_f32.irpa", file_type="irpa")
        model = VaeDecoderModel.from_dataset(ds).to(device="cpu", dtype=dtype)

        results = model.forward(inputs)
        torch.testing.assert_close(ref_results, results)

    @parameterized.expand(
        [
            (torch.float32, torch.float64, 1e-5, 1e-5),
            (torch.bfloat16, torch.float64, 3e-2, 3e-2),
        ],
    )
    @pytest.mark.xfail(
        platform.system() == "Windows",
        raises=AssertionError,
        strict=False,
        reason="nan on Windows",
    )
    def testCompareToyEagerVsHuggingFace(
        self,
        target_dtype: torch.dtype,
        reference_dtype: torch.dtype,
        atol: float,
        rtol: float,
    ):
        config = get_toy_vae_decoder_config()
        theta = make_vae_decoder_random_theta(config, dtype=target_dtype)
        model = VaeDecoderModel(config, theta)
        hf_model = convert_vae_decoder_to_hugging_face(model).to(dtype=reference_dtype)

        self.runTestCompareEagerVsHuggingFace(
            target_model=model, reference_model=hf_model, atol=atol, rtol=rtol
        )

    @parameterized.expand(
        [
            (torch.float32, torch.float64, 1e-5, 1e-5),
            (torch.bfloat16, torch.float64, 3e-2, 3e-2),
        ],
    )
    @pytest.mark.xfail(
        is_cpu_condition,
        raises=iree.compiler.CompilerToolError,
        strict=True,
        reason="Compiler error on CPU TODO: file issue",
    )
    def testCompareToyIreeVsEager(
        self,
        target_dtype: torch.dtype,
        reference_dtype: torch.dtype,
        atol: float,
        rtol: float,
    ):
        config = get_toy_vae_decoder_config()
        reference_theta = make_vae_decoder_random_theta(config, dtype=reference_dtype)
        reference_model = VaeDecoderModel(config, reference_theta)

        self.runTestCompareIreeVsEager(
            target_dtype=target_dtype,
            reference_model=reference_model,
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.expensive
    @with_vae_data
    def testVaeIreeVsHuggingFace(self):
        self.download_from_hf()
        self.import_bf16_dataset()
        self.import_f32_dataset()

        dtype = torch.bfloat16
        inputs = get_random_inputs(
            dtype=torch.float32, device="cpu", bs=1, config="flux"
        )
        ref_results = run_torch_vae(
            "black-forest-labs/FLUX.1-dev", inputs, 1024, 1024, True, torch.float32
        )

        ds = Dataset.load(f"{self._temp_dir}/flux_vae_bf16.irpa", file_type="irpa")
        ds_f32 = Dataset.load(f"{self._temp_dir}/flux_vae_f32.irpa", file_type="irpa")

        model = VaeDecoderModel.from_dataset(ds).to(device="cpu")
        model_f32 = VaeDecoderModel.from_dataset(ds_f32).to(device="cpu")

        # TODO: Decomposing attention due to https://github.com/iree-org/iree/issues/19286, remove once issue is resolved
        module = export_vae(model, inputs.to(dtype=dtype), True)
        module_f32 = export_vae(model_f32, inputs, True)

        module.save_mlir(f"{self._temp_dir}/flux_vae_bf16.mlir")
        module_f32.save_mlir(f"{self._temp_dir}/flux_vae_f32.mlir")

        iree.compiler.compile_file(
            f"{self._temp_dir}/flux_vae_bf16.mlir",
            output_file=f"{self._temp_dir}/flux_vae_bf16.vmfb",
            extra_args=self.extra_args,
        )
        iree.compiler.compile_file(
            f"{self._temp_dir}/flux_vae_f32.mlir",
            output_file=f"{self._temp_dir}/flux_vae_f32.vmfb",
            extra_args=self.extra_args,
        )

        iree_devices = get_iree_devices(driver=self.iree_device, device_count=1)

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=f"{self._temp_dir}/flux_vae_bf16.vmfb",
                devices=iree_devices,
                parameters_path=f"{self._temp_dir}/flux_vae_bf16.irpa",
            )

            input_args = OrderedDict([("inputs", inputs.to(dtype=dtype))])
            iree_args = flatten_for_iree_signature(input_args)

            iree_args = prepare_iree_module_function_args(
                args=iree_args, devices=iree_devices
            )
            iree_result = device_array_to_host(
                run_iree_module_function(
                    module=iree_module,
                    vm_context=iree_vm_context,
                    args=iree_args,
                    device=iree_devices[0],
                    function_name="decode",
                )[0]
            )

            # TODO verify these numerics
            torch.testing.assert_close(
                ref_results.to(torch.bfloat16), iree_result, atol=3.3e-2, rtol=4e5
            )

            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=f"{self._temp_dir}/flux_vae_f32.vmfb",
                devices=iree_devices,
                parameters_path=f"{self._temp_dir}/flux_vae_f32.irpa",
            )

            input_args = OrderedDict([("inputs", inputs)])
            iree_args = flatten_for_iree_signature(input_args)

            iree_args = prepare_iree_module_function_args(
                args=iree_args, devices=iree_devices
            )
            iree_result_f32 = device_array_to_host(
                run_iree_module_function(
                    module=iree_module,
                    vm_context=iree_vm_context,
                    args=iree_args,
                    device=iree_devices[0],
                    function_name="decode",
                )[0]
            )
            return iree_result_f32.clone()

        iree_result_f32 = with_iree_device_context(run_iree_module, iree_devices)

        torch.testing.assert_close(ref_results, iree_result_f32)

    def runTestCompareEagerVsHuggingFace(
        self,
        target_model: VaeDecoderModel,
        reference_model: AutoencoderKL,
        atol: float,
        rtol: float,
    ):
        inputs = get_random_inputs(
            dtype=target_model.dtype,
            device="cpu",
            bs=1,
            config="flux",
            height=target_model.hp.sample_size[0],
            width=target_model.hp.sample_size[1],
            latent_channels=target_model.hp.latent_channels,
        )
        ref_results = run_torch_vae(
            reference_model,
            inputs,
            height=target_model.hp.sample_size[0],
            width=target_model.hp.sample_size[1],
            flux=True,
        )
        results = target_model.forward(inputs)
        torch.testing.assert_close(
            results.to(dtype=ref_results.dtype), ref_results, atol=atol, rtol=rtol
        )

    def runTestCompareIreeVsEager(
        self,
        target_dtype: torch.dtype,
        reference_model: VaeDecoderModel,
        atol: float,
        rtol: float,
    ):
        target_theta = reference_model.theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        )
        target_model = VaeDecoderModel(reference_model.hp, theta=target_theta)

        target_inputs = get_random_inputs(
            dtype=target_model.dtype,
            device="cpu",
            bs=1,
            config="flux",
            height=target_model.hp.sample_size[0],
            width=target_model.hp.sample_size[1],
            latent_channels=target_model.hp.latent_channels,
        )
        reference_inputs = target_inputs.to(dtype=reference_model.dtype)

        reference_results = reference_model(reference_inputs)

        module = export_vae(target_model, target_inputs, True)
        target_mlir_path = f"{self._temp_dir}/model.mlir"
        target_module_path = f"{self._temp_dir}/model.vmfb"
        module.save_mlir(f"{self._temp_dir}/model.mlir")
        iree.compiler.compile_file(
            target_mlir_path,
            output_file=target_module_path,
            extra_args=self.extra_args,
        )
        target_parameters_path = f"{self._temp_dir}/model.irpa"
        target_dataset = Dataset(
            properties=target_model.hp.asdict(), root_theta=target_model.theta
        )
        target_dataset.save(target_parameters_path)

        iree_devices = get_iree_devices(driver=self.iree_device, device_count=1)

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=target_module_path,
                devices=iree_devices,
                parameters_path=target_parameters_path,
            )

            iree_args = flatten_for_iree_signature(target_inputs)

            iree_args = prepare_iree_module_function_args(
                args=iree_args, devices=iree_devices
            )
            target_results = device_array_to_host(
                run_iree_module_function(
                    module=iree_module,
                    vm_context=iree_vm_context,
                    args=iree_args,
                    device=iree_devices[0],
                    function_name="decode",
                )[0]
            ).to(dtype=reference_results.dtype)

            torch.testing.assert_close(
                reference_results, target_results, atol=atol, rtol=rtol
            )

        with_iree_device_context(run_iree_module, iree_devices)

    def download_from_hf(self):
        hf_hub_download(
            repo_id=self.hf_model_id,
            local_dir=f"{self._temp_dir}/flux_vae/",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/config.json",
        )
        hf_hub_download(
            repo_id=self.hf_model_id,
            local_dir=f"{self._temp_dir}/flux_vae/",
            local_dir_use_symlinks=False,
            revision="main",
            filename="vae/diffusion_pytorch_model.safetensors",
        )

    def import_bf16_dataset(self):
        dataset = import_hf_dataset(
            f"{self._temp_dir}/flux_vae/vae/config.json",
            [f"{self._temp_dir}/flux_vae/vae/diffusion_pytorch_model.safetensors"],
        )
        dataset.save(
            f"{self._temp_dir}/flux_vae_bf16.irpa", io_report_callback=logger.debug
        )

    def import_f32_dataset(self):
        dataset_f32 = import_hf_dataset(
            f"{self._temp_dir}/flux_vae/vae/config.json",
            [f"{self._temp_dir}/flux_vae/vae/diffusion_pytorch_model.safetensors"],
            target_dtype=torch.float32,
        )
        dataset_f32.save(
            f"{self._temp_dir}/flux_vae_f32.irpa", io_report_callback=logger.debug
        )


if __name__ == "__main__":
    unittest.main()
