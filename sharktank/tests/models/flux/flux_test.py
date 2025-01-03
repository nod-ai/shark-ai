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
from collections import OrderedDict
from sharktank.models.flux.export import (
    export_flux_transformer_from_hugging_face,
    export_flux_transformer,
)
from sharktank.models.flux.testing import (
    export_dev_random_single_layer,
    make_dev_single_layer_config,
    make_random_theta,
)
from sharktank.models.flux.flux import FluxModelV1
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

logging.basicConfig(level=logging.DEBUG)
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
    "--iree-codegen-llvmgpu-use-vector-distribution",
    "--iree-hip-waves-per-eu=2",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline,iree-preprocessing-pad-to-intrinsics)",
]


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
        self, reference_model: FluxModelV1, target_dtype: torch.dtype
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
        export_flux_transformer(
            target_torch_model,
            mlir_output_path=mlir_path,
            parameters_output_path=parameters_path,
            batch_sizes=batch_sizes,
        )

        iree_module_path = self._temp_dir / "model.vmfb"
        iree.compiler.compile_file(
            mlir_path,
            output_file=iree_module_path,
            extra_args=iree_compile_flags,
        )

        target_input_args, target_input_kwargs = target_torch_model.sample_inputs(
            batch_size
        )

        def covert_target_to_reference_dtype(t: torch.Tensor) -> torch.Tensor:
            if t.dtype == target_dtype:
                return t.to(dtype=reference_model.dtype)
            return t

        reference_input_args = [
            covert_target_to_reference_dtype(t) for t in target_input_args
        ]
        reference_input_kwargs = OrderedDict(
            (k, covert_target_to_reference_dtype(t))
            for k, t in target_input_kwargs.items()
        )

        reference_result_dict = call_torch_module_function(
            module=reference_model,
            function_name="forward",
            args=reference_input_args,
            kwargs=reference_input_kwargs,
        )
        expected_outputs = flatten_for_iree_signature(reference_result_dict)

        iree_devices = get_iree_devices(driver="hip", device_count=1)
        iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
            module_path=iree_module_path,
            devices=iree_devices,
            parameters_path=parameters_path,
        )
        iree_args = prepare_iree_module_function_args(
            args=flatten_for_iree_signature([target_input_args, target_input_kwargs]),
            devices=iree_devices,
        )

        iree_result = iree_to_torch(
            *run_iree_module_function(
                module=iree_module,
                vm_context=iree_vm_context,
                args=iree_args,
                driver="hip",
                function_name=f"forward_bs{batch_size}",
            )
        )
        actual_outputs = [
            ops.to(iree_result[i], dtype=expected_outputs[i].dtype)
            for i in range(len(expected_outputs))
        ]
        # TODO: figure out a good metric. Probably per pixel comparison would be good
        # enough.
        torch.testing.assert_close(actual_outputs, expected_outputs)

    def runCompareDevRandomSingleLayerIreeAgainstTorchEager(
        self, reference_dtype: torch.dtype, target_dtype: torch.dtype
    ):
        config = make_dev_single_layer_config()

        reference_theta = make_random_theta(config, reference_dtype)
        reference_theta.rename_tensors_to_paths()
        reference_model = FluxModelV1(
            theta=reference_theta,
            params=config,
        )
        self.runCompareIreeAgainstTorchEager(reference_model, target_dtype)

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="Accuracy is not good enough. The observed absolute error is 8976.53.",
    )
    @with_flux_data
    def testCompareDevRandomSingleLayerIreeBf16AgainstTorchEagerF32(self):
        self.runCompareDevRandomSingleLayerIreeAgainstTorchEager(
            reference_dtype=torch.float32, target_dtype=torch.bfloat16
        )

    @pytest.mark.xfail(
        raises=AssertionError,
        reason="Accuracy is probably not good enough. The observed absolute error is 73.25.",
    )
    @with_flux_data
    def testCompareDevRandomSingleLayerIreeF32AgainstTorchEagerF32(self):
        self.runCompareDevRandomSingleLayerIreeAgainstTorchEager(
            reference_dtype=torch.float32, target_dtype=torch.float32
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
