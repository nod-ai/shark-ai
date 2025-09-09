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
from diffusers import WanTransformer3DModel
from sharktank.layers import model_config_presets, create_model
from sharktank.models.wan.export import (
    export_wan_transformer_from_hugging_face,
    export_wan_transformer_model_mlir,
    import_wan_transformer_dataset_from_hugging_face,
)
from sharktank.models.wan.wan import WanModel, WanParams
from sharktank.models.wan.compile import iree_compile_flags
from sharktank.models.wan.tools.diffuser_ref import (
    convert_wan_transformer_input_for_hugging_face_model,
)
from sharktank.utils.testing import (
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
from sharktank.utils import chdir
from sharktank import ops
from sharktank.transforms.dataset import set_float_dtype
from sharktank.types import Dataset, Theta

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
with_wan_data = pytest.mark.skipif("not config.getoption('with_wan_data')")


def convert_dtype_if_dtype(
    t: torch.Tensor, source_dtype: torch.dtype, target_dtype: torch.dtype
) -> torch.Tensor:
    if t.dtype == source_dtype:
        return t.to(dtype=target_dtype)
    return t


@pytest.mark.usefixtures("path_prefix", "get_iree_flags")
class WanTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)

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

        mlir_path = self._temp_dir / "model.mlir"
        parameters_path = self._temp_dir / "parameters.irpa"
        batch_size = 1
        batch_sizes = [batch_size]
        logger.info("Exporting wan transformer to MLIR...")
        export_wan_transformer_model_mlir(
            target_torch_model,
            output_path=mlir_path,
            batch_sizes=batch_sizes,
        )

        iree_module_path = self._temp_dir / "model.vmfb"
        logger.info("Compiling MLIR file...")

        iree_device_flags = get_iree_compiler_flags_from_object(self)
        compile_flags = iree_compile_flags + iree_device_flags
        iree.compiler.compile_file(
            str(mlir_path),
            output_file=str(iree_module_path),
            extra_args=compile_flags,
        )

        reference_input_args, reference_input_kwargs = reference_model.sample_inputs(
            batch_size
        )
        assert len(reference_input_args) == 0

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
                args=flatten_for_iree_signature(reference_input_kwargs),
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
        torch.testing.assert_close(
            actual_outputs,
            expected_outputs,
            atol=atol,
            rtol=0,
            msg=f"Actual vs expected results diff > {atol}",
        )

    def runTestCompare1p3bIreeAgainstEager(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    ):
        parameters_output_path = self._temp_dir / "parameters.irpa"

        import_wan_transformer_dataset_from_hugging_face(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            parameters_output_path=parameters_output_path,
        )
        refrence_dataset = Dataset.load(parameters_output_path)
        refrence_dataset.root_theta = Theta(
            {
                k: set_float_dtype(t, reference_dtype)
                for k, t in refrence_dataset.root_theta.flatten().items()
            }
        )
        reference_model = WanModel(
            theta=refrence_dataset.root_theta,
            params=WanParams.from_hugging_face_properties(refrence_dataset.properties),
        )

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

        reference_input_kwargs = convert_wan_transformer_input_for_hugging_face_model(
            *reference_input_args, **reference_input_kwargs
        )

        reference_output = reference_model(**reference_input_kwargs)["sample"]
        target_output = target_model(*target_input_args, **target_input_kwargs)
        target_output = convert_dtype_if_dtype(
            target_output, source_dtype=target_model.dtype, target_dtype=reference_dtype
        )

        torch.testing.assert_close(
            target_output,
            reference_output,
            atol=atol,
            rtol=0,
            msg=f"Target and reference outputs differ > {atol}",
        )

    # def runTestCompareToyIreeAgainstEager(
    #     self, reference_dtype: torch.dtype, target_dtype: torch.dtype, atol: float
    # ):
    #     config = make_toy_config()
    #     reference_theta = make_random_theta(config, dtype=reference_dtype)
    #     reference_model = WanModel(theta=reference_theta, params=config)
    #     self.runCompareIreeAgainstTorchEager(
    #         reference_model=reference_model, target_dtype=target_dtype, atol=atol
    #     )

    # @pytest.mark.xfail(
    #     reason="Fails on both CPU and MI300. Issue: https://github.com/nod-ai/shark-ai/issues/1244",
    # )
    # def testCompareToyIreeF32AgainstEagerF64(self):
    #     """atol is apparently high because the expected output range is large.
    #     Its absolute maximum is 3915. Observed atol is 0.036."""
    #     self.runTestCompareToyIreeAgainstEager(
    #         reference_dtype=torch.float64, target_dtype=torch.float32, atol=1e-1
    #     )

    # @pytest.mark.xfail(
    #     reason="Fails on both CPU and MI300. Issue: https://github.com/nod-ai/shark-ai/issues/1244",
    # )
    # def testCompareToyIreeBf16AgainstEagerF64(self):
    #     """atol is apparently high because the expected output range is large.
    #     Its absolute maximum is 3915. Observed atol is 260.6.
    #     This is consistent with the expectation that bf16 atol should be worse by ~10^4
    #     compared to f32. f32 can represent ~7 digits and bf16 can represent ~3."""
    #     self.runTestCompareToyIreeAgainstEager(
    #         reference_dtype=torch.float64, target_dtype=torch.bfloat16, atol=5e2
    #     )

    @with_wan_data
    @pytest.mark.xfail(
        reason="Marking xfail with issue already present. Issue: https://github.com/nod-ai/shark-ai/issues/1244",
    )
    @pytest.mark.expensive
    def testCompare1p3bIreeF32AgainstEagerF32(self):
        self.runTestCompare1p3bIreeAgainstEager(
            reference_dtype=torch.float32, target_dtype=torch.float32, atol=1e-2
        )

    @with_wan_data
    @pytest.mark.expensive
    def testCompare1p3bIreeBf16AgainstEagerF32(self):
        self.runTestCompare1p3bIreeAgainstEager(
            reference_dtype=torch.float32, target_dtype=torch.bfloat16, atol=1
        )

    @with_wan_data
    # @pytest.mark.expensive
    def testCompare14bTorchEagerBf16AgainstHuggingFaceF32(self):
        parameters_output_path = self._temp_dir / "parameters.irpa"
        reference_dtype = torch.float32

        reference_model = WanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            subfolder="transformer",
            torch_dtype=reference_dtype,
        )

        import_wan_transformer_dataset_from_hugging_face(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            parameters_output_path=parameters_output_path,
        )
        target_dataset = Dataset.load(parameters_output_path)
        target_model = WanModel(
            theta=target_dataset.root_theta,
            params=WanParams.from_hugging_face_properties(target_dataset.properties),
        )

        self.runTestCompareTorchEagerAgainstHuggingFace(
            reference_model=reference_model,
            reference_dtype=reference_dtype,
            target_model=target_model,
            atol=4.0,
        )

    @with_wan_data
    @pytest.mark.expensive
    def testCompare1p3bTorchEagerF32AgainstHuggingFaceF32(self):
        parameters_output_path = self._temp_dir / "parameters.irpa"
        reference_dtype = torch.float32
        target_dtype = torch.float32

        reference_model = WanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B",
            subfolder="transformer",
            torch_dtype=reference_dtype,
        )

        import_wan_transformer_dataset_from_hugging_face(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            parameters_output_path=parameters_output_path,
        )
        target_dataset = Dataset.load(parameters_output_path)
        target_dataset.root_theta = Theta(
            {
                k: set_float_dtype(t, target_dtype)
                for k, t in target_dataset.root_theta.flatten().items()
            }
        )
        target_model = WanModel(
            theta=target_dataset.root_theta,
            params=WanParams.from_hugging_face_properties(target_dataset.properties),
        )

        self.runTestCompareTorchEagerAgainstHuggingFace(
            reference_model=reference_model,
            reference_dtype=reference_dtype,
            target_model=target_model,
            atol=1e-4,
        )

    @with_wan_data
    @pytest.mark.expensive
    def testExport1p3BTransformerFromHuggingFace(self):
        export_wan_transformer_from_hugging_face(
            "Wan-AI/Wan2.1-T2V-1.3B",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    @with_wan_data
    @pytest.mark.expensive
    def testExport14BTransformerFromHuggingFace(self):
        export_wan_transformer_from_hugging_face(
            "Wan-AI/Wan2.1-I2V-14B-480P",
            mlir_output_path=self._temp_dir / "model.mlir",
            parameters_output_path=self._temp_dir / "parameters.irpa",
        )

    # @with_wan_data
    # @pytest.mark.expensive
    # def testExportAndCompileFromPreset(self):
    #     with chdir(self._temp_dir):
    #         name = "WanAI-wan2-1-t2v-bf16-512x512-81f-hip-gfx942-release"
    #         config = model_config_presets[name]
    #         logger.info("Creating model...")
    #         model = create_model(config)
    #         logger.info("Exporting model...")
    #         model.export()
    #         logger.info("Compiling model...")
    #         model.compile()


if __name__ == "__main__":
    unittest.main()
