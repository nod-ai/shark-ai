# Copyright 2024 Advanced Micro T2Vices, Inc.
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
from diffusers import WanTransformer3DModel
from sharktank.models.wan.tools.export_all import export_component
from sharktank.models.wan.tools.compile_wan import get_compile_options, run_compilation

from sharktank.models.wan.testing import (
    convert_wan_transformer_input_for_hugging_face_model,
    export_wan_random_single_layer,
    make_toy_config,
    make_random_theta,
)
from sharktank.models.wan.wan import WanModel, WanParams
from sharktank.models.wan.export import (
    export_wan_transformer,
    import_wan_transformer_dataset_from_hugging_face,
)
from sharktank.utils.testing import (
    assert_cosine_similarity_close,
    TempDirTestBase,
)
from sharktank.utils.iree import (
    get_iree_compiler_flags_from_object,
    with_iree_device_context,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    flatten_for_iree_signature,
    iree_to_torch,
)
from sharktank.utils.logging import format_tensor_statistics
from sharktank import ops
from sharktank.transforms.dataset import set_float_dtype
from sharktank.types import Dataset, Theta, unbox_tensor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
with_wan_data = pytest.mark.skipif("not config.getoption('with_wan_data')")


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


wan_repo = "Wan-AI/Wan2.1-T2V-1.3B"
model_name = "wan2_1_1B"
dims = "512x512"
dtype = "bf16"
width = int(dims.split("x")[0])
height = int(dims.split("x")[1])
num_frames = 81


@pytest.mark.usefixtures("path_prefix", "iree_flags")
class WanTransformerTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    @pytest.mark.expensive
    def testExportWanRandomSingleLayerBf16(self):
        export_wan_random_single_layer(
            dtype=torch.bfloat16,
            batch_sizes=[1],
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    def runCompareIreeAgainstTorchEager(
        self,
        reference_model: WanModel,
        target_dtype: torch.dtype,
        atol: float,
    ):
        target_theta = reference_model.theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        )

        target_torch_model = WanModel(
            theta=target_theta,
            params=reference_model.params,
        )
        target_torch_model.set_export_config(height, width, num_frames)

        mlir_path = self._temp_dir / "model.mlir"
        parameters_path = self._temp_dir / "parameters.irpa"
        batch_size = 1
        batch_sizes = [batch_size]
        logger.info("Exporting wan transformer to MLIR...")
        export_wan_transformer(
            target_torch_model,
            mlir_output_path=mlir_path,
            parameters_output_path=parameters_path,
            batch_sizes=batch_sizes,
        )

        iree_module_path = self._temp_dir / "model.vmfb"
        logger.info("Compiling MLIR file...")

        iree_device_flags = get_iree_compiler_flags_from_object(self)
        _, iree_compile_flags = get_compile_options(
            "transformer", model_name, dims, dtype
        )
        compile_flags = iree_device_flags + iree_compile_flags["extra_args"]
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

        def run_iree_module(iree_devices: list[iree.runtime.HalT2Vice]):
            logger.info("Loading IREE module...")
            iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
                module_path=str(iree_module_path),
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
                unbox_tensor(ops.to(iree_result[i], dtype=expected_outputs[i].dtype))
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
        assert_cosine_similarity_close(
            actual_outputs[0],
            expected_outputs[0],
            atol=atol,
        )

    def runTestCompareT2VIreeAgainstEager(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    ):
        parameters_output_path = self._temp_dir / "parameters.irpa"

        import_wan_transformer_dataset_from_hugging_face(
            repo_id=wan_repo,
            parameters_output_path=parameters_output_path,
        )
        reference_dataset = Dataset.load(parameters_output_path)
        reference_dataset.root_theta = Theta(
            {
                k: set_float_dtype(t, reference_dtype)
                for k, t in reference_dataset.root_theta.flatten().items()
            }
        )
        reference_model = WanModel(
            theta=reference_dataset.root_theta,
            params=WanParams.from_hugging_face_properties(reference_dataset.properties),
        )
        reference_model.set_export_config(height, width, num_frames)

        self.runCompareIreeAgainstTorchEager(reference_model, target_dtype, atol=atol)

    def runTestCompareTorchEagerAgainstHuggingFace(
        self,
        reference_model: WanTransformer3DModel,
        reference_dtype: torch.dtype,
        target_model: WanModel,
        atol: float,
    ):
        target_input_args, target_input_kwargs = target_model.sample_inputs()

        assert len(target_input_args) == 0
        reference_input_args = []
        reference_input_kwargs = convert_input_dtype(
            target_input_kwargs, dtype=reference_dtype
        )

        reference_input_kwargs = convert_wan_transformer_input_for_hugging_face_model(
            *reference_input_args, **reference_input_kwargs
        )

        logger.info("Running reference model...")
        reference_output = reference_model(**reference_input_kwargs)["sample"]
        logger.info("Running target model...")
        target_output = target_model(*target_input_args, **target_input_kwargs)
        target_output = convert_dtype_if_dtype(
            target_output, source_dtype=target_model.dtype, target_dtype=reference_dtype
        )

        assert_cosine_similarity_close(
            target_output,
            reference_output,
            atol=atol,
        )

    def runTestCompareToyIreeAgainstEager(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    ):
        config = make_toy_config()
        reference_theta = make_random_theta(config, dtype=reference_dtype)
        reference_model = WanModel(theta=reference_theta, params=config)
        reference_model.set_export_config(height, width, num_frames)
        self.runCompareIreeAgainstTorchEager(
            reference_model=reference_model, target_dtype=target_dtype, atol=atol
        )

    def testCompareToyIreeF32AgainstEagerF64(self):
        self.runTestCompareToyIreeAgainstEager(
            reference_dtype=torch.float64, target_dtype=torch.float32, atol=1e-5
        )

    def testCompareToyIreeBf16AgainstEagerF64(self):
        self.runTestCompareToyIreeAgainstEager(
            reference_dtype=torch.float64, target_dtype=torch.bfloat16, atol=1e-3
        )

    @with_wan_data
    @pytest.mark.expensive
    def testCompareT2VIreeF32AgainstEagerF32(self):
        self.runTestCompareT2VIreeAgainstEager(
            reference_dtype=torch.float32, target_dtype=torch.float32, atol=1e-5
        )

    def testCompareT2VIreeBf16AgainstEagerF32(self):
        self.runTestCompareT2VIreeAgainstEager(
            reference_dtype=torch.float32, target_dtype=torch.bfloat16, atol=1e-3
        )

    @with_wan_data
    @pytest.mark.expensive
    def testCompareT2VTorchEagerBf16AgainstHuggingFaceF32(self):
        parameters_output_path = self._temp_dir / "parameters.irpa"
        reference_dtype = torch.float32

        reference_model = WanTransformer3DModel.from_pretrained(
            wan_repo,
            subfolder="transformer",
            torch_dtype=reference_dtype,
        )

        import_wan_transformer_dataset_from_hugging_face(
            repo_id=wan_repo,
            parameters_output_path=parameters_output_path,
        )
        target_dataset = Dataset.load(parameters_output_path)
        target_model = WanModel(
            theta=target_dataset.root_theta,
            params=WanParams.from_hugging_face_properties(target_dataset.properties),
        )
        target_model.set_export_config(height, width, num_frames)

        self.runTestCompareTorchEagerAgainstHuggingFace(
            reference_model=reference_model,
            reference_dtype=reference_dtype,
            target_model=target_model,
            atol=1e-3,
        )

    @with_wan_data
    @pytest.mark.expensive
    def testSmokeExportCompileWanTransformerFromHuggingFace(self):
        mlir_path, weights_path = export_component(
            component="transformer",
            height=height,
            width=width,
            num_frames=num_frames,
            wan_repo="wan-AI/Wan2.1-T2V-14B",
            batch_size=1,
            artifacts_path=self._temp_dir,
            return_paths=True,
        )
        _, compile_flags = get_compile_options("transformer", model_name, dims, dtype)
        vmfb_path = run_compilation(mlir_path, **compile_flags)


if __name__ == "__main__":
    unittest.main()
