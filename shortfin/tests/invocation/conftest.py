# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import urllib.request


@pytest.fixture(scope="session")
def mobilenet_onnx_path(tmp_path_factory):
    try:
        import onnx
    except ModuleNotFoundError:
        raise pytest.skip("onnx python package not available")
    parent_dir = tmp_path_factory.mktemp("mobilenet_onnx")
    onnx_path = parent_dir / "mobilenet.onnx"
    if not onnx_path.exists():
        print("Downloading mobilenet.onnx")
        urllib.request.urlretrieve(
            "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
            onnx_path,
        )
    return onnx_path


@pytest.fixture(scope="session")
def mobilenet_compiled_path(mobilenet_onnx_path, compile_flags):
    try:
        import iree.compiler.tools as tools
        import iree.compiler.tools.import_onnx.__main__ as import_onnx
    except ModuleNotFoundError:
        raise pytest.skip("iree.compiler packages not available")
    mlir_path = mobilenet_onnx_path.parent / "mobilenet.mlir"
    vmfb_path = mobilenet_onnx_path.parent / "mobilenet_cpu.vmfb"
    if not vmfb_path.exists():
        print("Compiling mobilenet")
        args = import_onnx.parse_arguments(
            ["-o", str(mlir_path), str(mobilenet_onnx_path), "--opset-version", "17"]
        )
        import_onnx.main(args)
        tools.compile_file(
            str(mlir_path),
            output_file=str(vmfb_path),
            input_type="onnx",
            extra_args=compile_flags,
        )
    return vmfb_path
