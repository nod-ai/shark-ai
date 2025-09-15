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
from sharktank.models.wan.tools.export_all import export_component
from sharktank.models.wan.tools.compile_wan import get_compile_options, run_compilation

# from sharktank.models.wan.testing import (
#     convert_wan_transformer_input_for_hugging_face_model,
#     export_dev_random_single_layer,
#     make_toy_config,
#     make_random_theta,
# )
from sharktank.models.wan.wan import WanModel, WanParams
from sharktank.utils.testing import (
    TempDirTestBase,
    skip,
    is_mi300x,
    is_cpu,
    is_cpu_condition,
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


model_name = "wan2_1"
dims = "512x512"
dtype = "bf16"
width = int(dims.split("x")[0])
height = int(dims.split("x")[1])
num_frames = 81


class WanTransformerTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

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


class WanCLIPTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    @with_wan_data
    @pytest.mark.expensive
    def testSmokeExportCompileWanCLIPRefModel(self):
        mlir_path, weights_path = export_component(
            component="clip",
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=1,
            artifacts_path=self._temp_dir,
            return_paths=True,
        )
        _, compile_flags = get_compile_options("clip", model_name, dims, dtype)
        vmfb_path = run_compilation(mlir_path, **compile_flags)


class WanUmt5xxlTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    @with_wan_data
    @pytest.mark.expensive
    @pytest.mark.xfail("Issue with vector distribution awaiting triage")
    def testSmokeExportCompileWanUmt5xxlModel(self):
        mlir_path, weights_path = export_component(
            component="t5",
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=1,
            artifacts_path=self._temp_dir,
            return_paths=True,
        )
        _, compile_flags = get_compile_options("t5", model_name, dims, dtype)
        vmfb_path = run_compilation(mlir_path, **compile_flags)


class WanVAETest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    @with_wan_data
    @pytest.mark.expensive
    @pytest.mark.xfail(
        "export issues past torch 2.5.1, issues with attention shape (head dim 384) in IREE rocm codegen https://github.com/iree-org/iree/issues/20804"
    )
    def testSmokeExportCompileWanVAERefModel(self):
        mlir_path, weights_path = export_component(
            component="vae",
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=1,
            artifacts_path=self._temp_dir,
            return_paths=True,
        )
        _, compile_flags = get_compile_options("vae", model_name, dims, dtype)
        vmfb_path = run_compilation(mlir_path, **compile_flags)


if __name__ == "__main__":
    unittest.main()
