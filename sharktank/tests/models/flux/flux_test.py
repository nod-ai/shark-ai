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
from sharktank.models.flux.export import (
    export_flux_transformer_from_hugging_face,
    export_flux_transformer,
    import_flux_transformer_dataset_from_hugging_face,
)
from sharktank.models.flux.testing import (
    convert_flux_transformer_input_for_hugging_face_model,
    export_dev_random_single_layer,
    make_dev_single_layer_config,
    make_random_theta,
)
from sharktank.models.flux.flux import FluxModelV1, FluxParams
from sharktank.utils.testing import TempDirTestBase
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
from sharktank.transforms.dataset import set_float_dtype
from sharktank.types import Dataset, Theta

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
with_flux_data = pytest.mark.skipif("not config.getoption('with_flux_data')")

iree_compile_flags = [
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
    "--iree-codegen-llvmgpu-use-vector-distribution=1",
    "--iree-hip-waves-per-eu=2",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics,util.func(iree-preprocessing-generalize-linalg-matmul-experimental))",
]


def convert_dtype_if_dtype(
    t: torch.Tensor, source_dtype: torch.dtype, target_dtype: torch.dtype
) -> torch.Tensor:
    if t.dtype == source_dtype:
        return t.to(dtype=target_dtype)
    return t


class FluxTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)

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
        iree.compiler.compile_file(
            str(mlir_path),
            output_file=str(iree_module_path),
            extra_args=iree_compile_flags,
        )

        target_input_args, target_input_kwargs = target_torch_model.sample_inputs(
            batch_size
        )

        reference_input_args = [
            convert_dtype_if_dtype(
                t, source_dtype=target_dtype, target_dtype=reference_model.dtype
            )
            for t in target_input_args
        ]
        reference_input_kwargs = OrderedDict(
            (
                k,
                convert_dtype_if_dtype(
                    t, source_dtype=target_dtype, target_dtype=reference_model.dtype
                ),
            )
            for k, t in target_input_kwargs.items()
        )

        logger.info("Invoking reference torch function...")
        reference_result_dict = call_torch_module_function(
            module=reference_model,
            function_name="forward",
            args=reference_input_args,
            kwargs=reference_input_kwargs,
        )
        expected_outputs = flatten_for_iree_signature(reference_result_dict)

        iree_devices = get_iree_devices(driver="hip", device_count=1)
        logger.info("Loading IREE module...")
        iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
            module_path=iree_module_path,
            devices=iree_devices,
            parameters_path=parameters_path,
        )
        iree_args = prepare_iree_module_function_args(
            args=flatten_for_iree_signature([target_input_args, target_input_kwargs]),
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
        logger.info("Comparing outputs...")
        torch.testing.assert_close(actual_outputs, expected_outputs, atol=atol, rtol=0)

    def runTestCompareDevIreeAgainstHuggingFace(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    ):
        parameters_output_path = self._temp_dir / "parameters.irpa"

        import_flux_transformer_dataset_from_hugging_face(
            repo_id="black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
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

        reference_input_args = [
            convert_dtype_if_dtype(
                t, source_dtype=target_model.dtype, target_dtype=reference_dtype
            )
            for t in target_input_args
        ]
        reference_input_kwargs = OrderedDict(
            (
                k,
                convert_dtype_if_dtype(
                    t, source_dtype=target_model.dtype, target_dtype=reference_dtype
                ),
            )
            for k, t in target_input_kwargs.items()
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

    @with_flux_data
    def testCompareDevIreeF32AgainstHuggingFaceF32(self):
        self.runTestCompareDevIreeAgainstHuggingFace(
            reference_dtype=torch.float32, target_dtype=torch.float32, atol=1e-2
        )

    @pytest.mark.skip(
        reason="Segmentation fault during output comparison. See https://github.com/nod-ai/shark-ai/issues/1050"
    )
    @with_flux_data
    def testCompareDevIreeBf16AgainstHuggingFaceF32(self):
        self.runTestCompareDevIreeAgainstHuggingFace(
            reference_dtype=torch.float32, target_dtype=torch.bfloat16, atol=1
        )

    @with_flux_data
    def testCompareDevTorchEagerBf16AgainstHuggingFaceF32(self):
        parameters_output_path = self._temp_dir / "parameters.irpa"
        reference_dtype = torch.float32

        reference_model = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=reference_dtype,
        )

        import_flux_transformer_dataset_from_hugging_face(
            repo_id="black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
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
            repo_id="black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
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
    def testExportSchnellTransformerFromHuggingFace(self):
        export_flux_transformer_from_hugging_face(
            "black-forest-labs/FLUX.1-schnell/black-forest-labs-transformer",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    @with_flux_data
    def testExportDevTransformerFromHuggingFace(self):
        export_flux_transformer_from_hugging_face(
            "black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )


if __name__ == "__main__":
    unittest.main()
