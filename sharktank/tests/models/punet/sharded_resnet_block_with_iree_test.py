# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.compiler
import iree.runtime
import pytest
import sys
import torch

from pathlib import Path
from amdsharktank import ops
from amdsharktank.models.punet.testing import (
    export_sharded_toy_resnet_block_iree_test_data,
)
from amdsharktank.types import *
from amdsharktank.utils.iree import (
    get_iree_devices,
    iree_to_torch,
    load_iree_module,
    prepare_iree_module_function_args,
    run_iree_module_function,
    with_iree_device_context,
)
from typing import List


def get_compiler_args(target_device_kind: str, shard_count: int) -> List[str]:
    result = [
        f"--iree-hal-target-device={target_device_kind}[{i}]"
        for i in range(shard_count)
    ]
    result += [
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ]
    return result


def compile_iree_module(mlir_path: str, module_path: str, shard_count: int):
    iree.compiler.compile_file(
        mlir_path,
        output_file=module_path,
        extra_args=get_compiler_args(
            target_device_kind="local", shard_count=shard_count
        ),
    )


def run_test_toy_size_sharded_resnet_block_with_iree(artifacts_dir: Path):
    mlir_path = artifacts_dir / "model.mlir"
    module_path = artifacts_dir / "model.vmfb"
    parameters_path = artifacts_dir / "model.irpa"
    input_args_path = artifacts_dir / "input_args.irpa"
    expected_results_path = artifacts_dir / "expected_results.irpa"

    target_dtype = torch.float32
    shard_count = 2
    export_sharded_toy_resnet_block_iree_test_data(
        mlir_path=mlir_path,
        parameters_path=parameters_path,
        input_args_path=input_args_path,
        expected_results_path=expected_results_path,
        target_dtype=target_dtype,
        shard_count=shard_count,
    )

    compile_iree_module(
        mlir_path=str(mlir_path),
        module_path=str(module_path),
        shard_count=shard_count,
    )

    input_args = Dataset.load(input_args_path).root_theta.flatten()
    expected_results = Dataset.load(expected_results_path).root_theta.flatten()

    input_args = [input_args[f"{i}"] for i in range(len(input_args))]
    expected_results = [
        unbox_tensor(expected_results[f"{i}"]) for i in range(len(expected_results))
    ]

    iree_devices = get_iree_devices(driver="local-task", device_count=shard_count)

    def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
        iree_module, iree_vm_context, iree_vm_instance = load_iree_module(
            module_path=module_path,
            devices=iree_devices,
            parameters_path=parameters_path,
            tensor_parallel_size=shard_count,
        )
        iree_args = prepare_iree_module_function_args(
            args=input_args, devices=iree_devices
        )
        iree_results = iree_to_torch(
            *run_iree_module_function(
                module=iree_module,
                vm_context=iree_vm_context,
                args=iree_args,
                device=iree_devices[0],
                function_name="forward",
            )
        )
        return [
            ops.to(iree_results[i], dtype=expected_results[i].dtype).clone()
            for i in range(len(expected_results))
        ]

    actual_outputs = with_iree_device_context(run_iree_module, iree_devices)

    # Observed atol is 1e-5.
    torch.testing.assert_close(actual_outputs, expected_results, rtol=0, atol=5e-5)


@pytest.mark.xfail(
    condition=(sys.platform != "win32"),
    raises=iree.compiler.CompilerToolError,
    reason=(
        "The compiler crashes. "
        "See IREE issue https://github.com/iree-org/iree/issues/21049. "
        "It also failed once with "
        "Error code: -6 "
        "Diagnostics: "
        "malloc(): unsorted double linked list corrupted"
    ),
)
@pytest.mark.xfail(
    condition=(sys.platform == "win32"),
    raises=iree.compiler.CompilerToolError,
    reason=(
        "Used to fail with "
        "compiler error: operation's operand is unlinked. "
        "See https://github.com/iree-org/iree/issues/21114. "
        "Now it fails with error: 'stream.async.execute' op operand #0 must be "
        "variadic of resource or external resource or transient resource or variable "
        "resource or constant resource or staging resource, but got "
        "'!stream.timepoint'"
    ),
)
def test_toy_size_sharded_resnet_block_with_iree(tmp_path: Path):
    """Test sharding, exportation and execution with IREE local-task of a Resnet block.
    The result is compared against execution with torch.
    The model is tensor sharded across 2 devices.
    """

    run_test_toy_size_sharded_resnet_block_with_iree(artifacts_dir=tmp_path)
