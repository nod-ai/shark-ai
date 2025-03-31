# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys

from pathlib import Path
import tempfile

import torch


from iree.turbine import aot
from sharktank.models.punet.testing import make_resnet_block_2d_theta
from sharktank.models.punet.layers import ResnetBlock2D
from sharktank.models.punet.sharding import ResnetBlock2DSplitOutputChannelsSharding
from sharktank import ops
from sharktank.types import *
from sharktank.utils.iree import with_iree_device_context
import iree.runtime
from typing import List, Optional
import os

vm_context: iree.runtime.VmContext = None


def get_compiler_args(target_device_kind: str, shard_count: int) -> List[str]:
    result = [
        f"--iree-hal-target-device={target_device_kind}[{i}]"
        for i in range(shard_count)
    ]
    if target_device_kind == "local":
        result.append("--iree-hal-local-target-device-backends=llvm-cpu")
    return result


def compile_iree_module(
    export_output: aot.ExportOutput, module_path: str, shard_count: int
):
    export_output.session.set_flags(
        *get_compiler_args(target_device_kind="local", shard_count=shard_count)
    )
    export_output.compile(save_to=module_path, target_backends=None)


# TODO: improve IREE's Python API to be more concise in a multi-device context.
# This run function should be way shorter.
def run_iree_module(
    sharded_input_image: ShardedTensor,
    sharded_input_time_emb: ShardedTensor,
    module_path: str,
    parameters_path: str,
) -> ShardedTensor:
    assert sharded_input_image.shard_count == sharded_input_time_emb.shard_count
    shard_count = sharded_input_image.shard_count
    hal_driver = iree.runtime.get_driver("local-task")
    vm_instance = iree.runtime.VmInstance()
    available_devices = hal_driver.query_available_devices()
    # Use the same actual device for all devices.
    devices = [
        hal_driver.create_device(available_devices[0]) for _ in range(shard_count)
    ]

    def run_iree_module(devices: list[iree.runtime.HalDevice]):
        hal_module = iree.runtime.create_hal_module(
            instance=vm_instance, devices=devices
        )
        params_path = Path(parameters_path)
        # TODO: make IREE able to load the parameters from the top parameter file
        # without having to specify the parameter file for each shard separately.
        parameter_index = iree.runtime.ParameterIndex()
        for i in range(shard_count):
            parameter_index.load(
                file_path=str(
                    Path(params_path).with_suffix(f".rank{i}{params_path.suffix}")
                )
            )
        parameter_provider = parameter_index.create_provider(scope="model")
        parameters_module = iree.runtime.create_io_parameters_module(
            vm_instance, parameter_provider
        )

        vm_module = iree.runtime.VmModule.mmap(vm_instance, str(module_path))

        # The context needs to be destroyed after the buffers, although
        # it is not associate with them on the API level.
        global vm_context
        vm_context = iree.runtime.VmContext(
            instance=vm_instance, modules=(hal_module, parameters_module, vm_module)
        )
        module_input_args = [
            iree.runtime.asdevicearray(
                devices[i], sharded_input_image.shards[i].as_torch().to("cpu").numpy()
            )
            for i in range(shard_count)
        ]
        module_input_args += [
            iree.runtime.asdevicearray(
                devices[i],
                sharded_input_time_emb.shards[i].as_torch().to("cpu").numpy(),
            )
            for i in range(shard_count)
        ]

        vm_function = vm_module.lookup_function("main")
        invoker = iree.runtime.FunctionInvoker(
            vm_context=vm_context,
            # TODO: rework iree.runtime.FunctionInvoker interface for multiple devices.
            # This works, but does not look right.
            device=devices[0],
            vm_function=vm_function,
        )
        results = invoker(*module_input_args)
        shards = [torch.tensor(tensor.to_host()).clone() for tensor in results]
        return SplitPrimitiveTensor(ts=shards, shard_dim=1)

    return with_iree_device_context(run_iree_module, devices)


def run_test_sharded_resnet_block_with_iree(
    mlir_path: Path, module_path: Path, parameters_path: Path, caching: bool
):
    torch.set_default_dtype(torch.float32)
    batches = 2
    in_channels = 6
    out_channels = [12, 8]
    height = 11
    width = 13
    kernel_height = 5
    kernel_width = 5
    input_time_emb_shape = [batches, 8]
    norm_groups = 2
    eps = 0.01
    shard_count = 2

    torch.manual_seed(123456)

    input_image = torch.rand(
        batches,
        in_channels,
        height,
        width,
    )
    input_time_emb = torch.rand(input_time_emb_shape)

    unsharded_theta = make_resnet_block_2d_theta(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        input_time_emb_channels=input_time_emb_shape[1],
    )
    unsharded_theta.rename_tensors_to_paths()

    if not caching or not os.path.exists(parameters_path):
        sharding_spec = ResnetBlock2DSplitOutputChannelsSharding(
            shard_count=shard_count
        )
        sharded_theta = ops.reshard(unsharded_theta, sharding_spec)

        # Roundtrip the dataset, which anchors the tensors as parameters to be loaded
        # vs constants to be frozen (TODO: This is a bit wonky).
        sharded_dataset = Dataset({}, sharded_theta)
        sharded_dataset.save(parameters_path)

    sharded_dataset = Dataset.load(parameters_path)

    sharded_resnet_block = ResnetBlock2D(
        theta=sharded_dataset.root_theta,
        groups=norm_groups,
        eps=eps,
        non_linearity="relu",
        output_scale_factor=None,
        dropout=0.0,
        temb_channels=input_time_emb_shape[1],
        time_embedding_norm="default",
    )
    sharded_input_image = ops.reshard_split(input_image, dim=1, count=shard_count)
    sharded_input_time_emb = ops.replicate(input_time_emb, count=shard_count)
    expected_result = sharded_resnet_block(sharded_input_image, sharded_input_time_emb)

    # Verify as a sanity check that the sharded torch model matches the result
    # of the unsharded torch model.
    unsharded_resnet_block = ResnetBlock2D(
        theta=unsharded_theta,
        groups=norm_groups,
        eps=eps,
        non_linearity="relu",
        output_scale_factor=None,
        dropout=0.0,
        temb_channels=input_time_emb_shape[1],
        time_embedding_norm="default",
    )
    unsharded_result = unsharded_resnet_block(input_image, input_time_emb)
    torch.testing.assert_close(unsharded_result, ops.unshard(expected_result))

    if not caching or not os.path.exists(module_path):
        exported_resnet_block = aot.export(
            sharded_resnet_block,
            args=(
                sharded_input_image,
                sharded_input_time_emb,
            ),
        )
        exported_resnet_block.save_mlir(mlir_path)

        compile_iree_module(
            export_output=exported_resnet_block,
            module_path=module_path,
            shard_count=shard_count,
        )

    actual_result = run_iree_module(
        sharded_input_image=sharded_input_image,
        sharded_input_time_emb=sharded_input_time_emb,
        module_path=module_path,
        parameters_path=parameters_path,
    )
    assert len(actual_result.shards) == len(expected_result.shards)
    # TODO: reenable this test once numerical issues are resolved.
    # The absolute accuracy is > 0.00042. Is this good enough?
    # Maybe add a test with fp64, where if the accuracy is high would give us more
    # confidence that fp32 is also OK.
    for actual_shard, expected_shard in zip(
        actual_result.shards, expected_result.shards
    ):
        torch.testing.assert_close(
            unbox_tensor(actual_shard), unbox_tensor(expected_shard)
        )

    global vm_context
    del vm_context


@pytest.mark.xfail(
    reason="Maybe numerical issues with low accuracy.",
    strict=True,
    raises=AssertionError,
)
@pytest.mark.xfail(
    torch.__version__ >= (2, 5), reason="https://github.com/nod-ai/shark-ai/issues/683"
)
@pytest.mark.skipif(
    sys.platform == "win32", reason="https://github.com/nod-ai/shark-ai/issues/698"
)
def test_sharded_resnet_block_with_iree(
    mlir_path: Optional[Path],
    module_path: Optional[Path],
    parameters_path: Optional[Path],
    caching: bool,
):
    """Test sharding, exportation and execution with IREE local-task of a Resnet block.
    The result is compared against execution with torch.
    The model is tensor sharded across 2 devices.
    """

    with tempfile.TemporaryDirectory(
        # TODO: verify hypothesis and remove ignore_cleanup_errors=True
        # torch.export.export is spawning some processes that don't exit when the
        # function returns, this causes some objects to not get destroyed, which
        # in turn holds files params.rank0.irpa and params.rank1.irpa open.
        ignore_cleanup_errors=True
    ) as tmp_dir:
        mlir_path = Path(tmp_dir) / "model.mlir" if mlir_path is None else mlir_path
        module_path = (
            Path(tmp_dir) / "module.vmfb" if module_path is None else module_path
        )
        parameters_path = (
            Path(tmp_dir) / "params.irpa"
            if parameters_path is None
            else parameters_path
        )
        run_test_sharded_resnet_block_with_iree(
            mlir_path, module_path, parameters_path, caching
        )
