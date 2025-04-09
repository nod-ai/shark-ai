# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import functools
import unittest
import torch
import pytest
import iree.compiler
import iree.runtime
from collections import OrderedDict
from diffusers import FluxTransformer2DModel
from sharktank.layers import model_config_presets, create_model
from sharktank.models.flux.export import (
    export_flux_transformer_from_hugging_face,
    export_flux_transformer,
    import_flux_transformer_dataset_from_hugging_face,
)
from sharktank.models.flux.testing import (
    convert_flux_transformer_input_for_hugging_face_model,
    export_dev_random_single_layer,
    make_toy_config,
    make_random_theta,
)
from sharktank.models.flux.flux import FluxModelV1, FluxParams
from sharktank.models.flux.compile import iree_compile_flags
from sharktank.utils.testing import (
    TempDirTestBase,
    skip,
    is_mi300x,
    is_cpu,
    is_cpu_condition,
)
from sharktank.utils.iree import (
    with_iree_device_context,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
from sharktank.utils.logging import format_tensor_statistics
from sharktank.utils import chdir
from sharktank import ops
from sharktank.transforms.dataset import set_float_dtype
from sharktank.types import Dataset, Theta

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
with_flux_data = pytest.mark.skipif("not config.getoption('with_flux_data')")


def convert_dtype_if_dtype(
    t: torch.Tensor, source_dtype: torch.dtype, target_dtype: torch.dtype
) -> torch.Tensor:
    if t.dtype == source_dtype:
        return t.to(dtype=target_dtype)
    return t


def convert_input_dtype(input: dict[str, torch.Tensor], dtype: torch.dtype):
    always_float32_input_arg_names = set(["img_ids", "txt_ids"])
    return OrderedDict(
        (k, t if k in always_float32_input_arg_names else t.to(dtype=dtype))
        for k, t in input.items()
    )


@pytest.mark.usefixtures("path_prefix", "get_iree_flags")
class FluxTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)

    @pytest.mark.expensive
    def testExportDevRandomSingleLayerBf16(self):
        export_dev_random_single_layer(
            dtype=torch.bfloat16,
            batch_sizes=[1],
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    def runCompareIreeAgainstTorchEager(
        self,
        reference_model: FluxModelV1,
        target_dtype: torch.dtype,
        atol: float,
    ):
        target_theta = reference_model.theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        )

        target_torch_model = FluxModelV1(
            theta=target_theta,
            params=reference_model.params,
        )

        mlir_path = self._temp_dir / "model.mlir"
        parameters_path = self._temp_dir / "parameters.irpa"
        batch_size = 1
        batch_sizes = [batch_size]
        logger.info("Exporting flux transformer to MLIR...")
        export_flux_transformer(
            target_torch_model,
            mlir_output_path=mlir_path,
            parameters_output_path=parameters_path,
            batch_sizes=batch_sizes,
        )

        iree_module_path = self._temp_dir / "model.vmfb"
        logger.info("Compiling MLIR file...")
        compile_flags = iree_compile_flags + [
            f"--iree-hal-target-device={self.iree_hal_target_device}",
            f"--iree-hip-target={self.iree_hip_target}",
        ]
        iree.compiler.compile_file(
            str(mlir_path),
            output_file=str(iree_module_path),
            extra_args=compile_flags,
        )

        reference_input_args, reference_input_kwargs = reference_model.sample_inputs(
            batch_size
        )
        assert len(reference_input_args) == 0
        target_input_kwargs = convert_input_dtype(
            reference_input_kwargs, dtype=target_dtype
        )

        logger.info("Invoking reference torch function...")
        reference_result_dict = call_torch_module_function(
            module=reference_model,
            function_name="forward",
            args=reference_input_args,
            kwargs=reference_input_kwargs,
        )
        expected_outputs = flatten_for_iree_signature(reference_result_dict)

        iree_devices = [iree.runtime.get_device(self.iree_device, cache=False)]

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            logger.info("Loading IREE module...")
            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=iree_module_path,
                devices=iree_devices,
                parameters_path=parameters_path,
            )
            iree_args = prepare_iree_module_function_args(
                args=flatten_for_iree_signature(target_input_kwargs),
                devices=iree_devices,
            )

            logger.info("Invoking IREE function...")
            iree_result = iree_to_torch(
                *run_iree_module_function(
                    module=iree_module,
                    vm_context=iree_vm_context,
                    args=iree_args,
                    device=iree_devices[0],
                    function_name=f"forward_bs{batch_size}",
                )
            )
            actual_outputs = [
                ops.to(iree_result[i], dtype=expected_outputs[i].dtype)
                for i in range(len(expected_outputs))
            ]
            return [t.clone() for t in actual_outputs]

        actual_outputs = with_iree_device_context(run_iree_module, iree_devices)

        logger.info("Comparing outputs...")
        logger.info(f"Expected output {format_tensor_statistics(expected_outputs[0])}")
        abs_diff = (actual_outputs[0] - expected_outputs[0]).abs()
        logger.info(
            f"Actual vs expected abs diff {format_tensor_statistics(abs_diff[0])}"
        )
        torch.testing.assert_close(actual_outputs, expected_outputs, atol=atol, rtol=0)

    def runTestCompareDevIreeAgainstEager(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    ):
        parameters_output_path = self._temp_dir / "parameters.irpa"

        import_flux_transformer_dataset_from_hugging_face(
            repo_id="black-forest-labs/FLUX.1-dev",
            parameters_output_path=parameters_output_path,
        )
        refrence_dataset = Dataset.load(parameters_output_path)
        refrence_dataset.root_theta = Theta(
            {
                k: set_float_dtype(t, reference_dtype)
                for k, t in refrence_dataset.root_theta.flatten().items()
            }
        )
        reference_model = FluxModelV1(
            theta=refrence_dataset.root_theta,
            params=FluxParams.from_hugging_face_properties(refrence_dataset.properties),
        )

        self.runCompareIreeAgainstTorchEager(reference_model, target_dtype, atol=atol)

    def runTestCompareTorchEagerAgainstHuggingFace(
        self,
        reference_model: FluxTransformer2DModel,
        reference_dtype: torch.dtype,
        target_model: FluxModelV1,
        atol: float,
    ):
        target_input_args, target_input_kwargs = target_model.sample_inputs()

        assert len(target_input_args) == 0
        reference_input_args = []
        reference_input_kwargs = convert_input_dtype(
            target_input_kwargs, dtype=reference_dtype
        )

        reference_input_kwargs = convert_flux_transformer_input_for_hugging_face_model(
            *reference_input_args, **reference_input_kwargs
        )

        reference_output = reference_model(**reference_input_kwargs)["sample"]
        target_output = target_model(*target_input_args, **target_input_kwargs)
        target_output = convert_dtype_if_dtype(
            target_output, source_dtype=target_model.dtype, target_dtype=reference_dtype
        )

        torch.testing.assert_close(target_output, reference_output, atol=atol, rtol=0)

    def runTestCompareToyIreeAgainstEager(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    ):
        config = make_toy_config()
        reference_theta = make_random_theta(config, dtype=reference_dtype)
        reference_model = FluxModelV1(theta=reference_theta, params=config)
        self.runCompareIreeAgainstTorchEager(
            reference_model=reference_model, target_dtype=target_dtype, atol=atol
        )

    @pytest.mark.xfail(
        is_cpu_condition,
        raises=iree.compiler.CompilerToolError,
        strict=True,
        reason="The compiler segfaults https://github.com/iree-org/iree/issues/20283",
    )
    def testCompareToyIreeF32AgainstEagerF64(self):
        """atol is apparently high because the expected output range is large.
        Its absolute maximum is 3915. Observed atol is 0.036."""
        self.runTestCompareToyIreeAgainstEager(
            reference_dtype=torch.float64, target_dtype=torch.float32, atol=1e-1
        )

    @pytest.mark.xfail(
        is_cpu_condition,
        raises=iree.compiler.CompilerToolError,
        strict=True,
        reason="The compiler segfaults https://github.com/iree-org/iree/issues/20283",
    )
    def testCompareToyIreeBf16AgainstEagerF64(self):
        """atol is apparently high because the expected output range is large.
        Its absolute maximum is 3915. Observed atol is 260.6.
        This is consistent with the expectation that bf16 atol should be worse by ~10^4
        compared to f32. f32 can represent ~7 digits and bf16 can represent ~3."""
        self.runTestCompareToyIreeAgainstEager(
            reference_dtype=torch.float64, target_dtype=torch.bfloat16, atol=5e2
        )

    @with_flux_data
    @pytest.mark.expensive
    def testCompareDevIreeF32AgainstEagerF32(self):
        self.runTestCompareDevIreeAgainstEager(
            reference_dtype=torch.float32, target_dtype=torch.float32, atol=1e-2
        )

    @with_flux_data
    @pytest.mark.expensive
    def testCompareDevIreeBf16AgainstEagerF32(self):
        self.runTestCompareDevIreeAgainstEager(
            reference_dtype=torch.float32, target_dtype=torch.bfloat16, atol=1
        )

    @with_flux_data
    @pytest.mark.expensive
    def testCompareDevTorchEagerBf16AgainstHuggingFaceF32(self):
        parameters_output_path = self._temp_dir / "parameters.irpa"
        reference_dtype = torch.float32

        reference_model = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=reference_dtype,
        )

        import_flux_transformer_dataset_from_hugging_face(
            repo_id="black-forest-labs/FLUX.1-dev",
            parameters_output_path=parameters_output_path,
        )
        target_dataset = Dataset.load(parameters_output_path)
        target_model = FluxModelV1(
            theta=target_dataset.root_theta,
            params=FluxParams.from_hugging_face_properties(target_dataset.properties),
        )

        self.runTestCompareTorchEagerAgainstHuggingFace(
            reference_model=reference_model,
            reference_dtype=reference_dtype,
            target_model=target_model,
            atol=4.0,
        )

    @with_flux_data
    @pytest.mark.expensive
    def testCompareDevTorchEagerF32AgainstHuggingFaceF32(self):
        parameters_output_path = self._temp_dir / "parameters.irpa"
        reference_dtype = torch.float32
        target_dtype = torch.float32

        reference_model = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=reference_dtype,
        )

        import_flux_transformer_dataset_from_hugging_face(
            repo_id="black-forest-labs/FLUX.1-dev",
            parameters_output_path=parameters_output_path,
        )
        target_dataset = Dataset.load(parameters_output_path)
        target_dataset.root_theta = Theta(
            {
                k: set_float_dtype(t, target_dtype)
                for k, t in target_dataset.root_theta.flatten().items()
            }
        )
        target_model = FluxModelV1(
            theta=target_dataset.root_theta,
            params=FluxParams.from_hugging_face_properties(target_dataset.properties),
        )

        self.runTestCompareTorchEagerAgainstHuggingFace(
            reference_model=reference_model,
            reference_dtype=reference_dtype,
            target_model=target_model,
            atol=1e-4,
        )

    @with_flux_data
    @pytest.mark.expensive
    def testExportSchnellTransformerFromHuggingFace(self):
        export_flux_transformer_from_hugging_face(
            "black-forest-labs/FLUX.1-schnell",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    @with_flux_data
    @pytest.mark.expensive
    def testExportDevTransformerFromHuggingFace(self):
        export_flux_transformer_from_hugging_face(
            "black-forest-labs/FLUX.1-dev",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    @with_flux_data
    @pytest.mark.expensive
    def testExportAndCompileFromPreset(self):
        with chdir(self._temp_dir):
            name = "black-forest-labs--FLUX.1-dev-bf16-1024x1024-hip-gfx942-release"
            config = model_config_presets[name]
            logger.info("Creating model...")
            model = create_model(config)
            logger.info("Exporting model...")
            model.export()
            logger.info("Compiling model...")
            model.compile()


if __name__ == "__main__":
    unittest.main()
