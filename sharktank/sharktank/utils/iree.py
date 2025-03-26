# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime
from typing import Any, Callable, Generator, List, Tuple, Optional, Union
from pathlib import Path
import torch
import os
import numpy as np
import collections.abc
from collections import OrderedDict
from contextlib import contextmanager
import gc
from ..types.tensors import (
    AnyTensor,
    InferenceTensor,
    ShardedTensor,
    DefaultPrimitiveTensor,
    unbox_tensor,
    torch_tree_flatten,
)
from .tree import Tree


def with_iree_device_context(
    fn: Callable[[list[iree.runtime.HalDevice]], Any],
    devices: list[iree.runtime.HalDevice],
):
    """Run a function with the provided devices and make sure all local resources
    created in the function are cleaned up.

    This construct is required as iree.runtime.HalBuffer, iree.runtime.HalBufferView
    and iree.runtime.MappedMemory do not hold a reference to their respective
    HalDevice, but they must be destroyed before the device is destroyed.
    They are thin wrappers of the underlying native objects and they do not hold
    references to their parent devices to avoid circular references.
    To ensure a correct destruction order it is desirable that callable argument does
    not return or leak arrays to the external context that are backed by IREE native
    buffers.
    If that is the case the user is responsible for destruction order.

    An example usage that may cause a problem is
    ```
    def f():
        dev: iree.runtime.HalDevice = ...
        dev_arr: iree.runtime.DeviceArray = ...

        # This creates a numpy array that is backed by iree.runtime.MappedMemory.
        arr = dev_arr.to_host()

        del dev_arr

        t = torch.tensor(arr)
    ```
    Although the dev variable will be deleted after all other variables, in practice
    with the various object wrappings with numpy and torch, the underlying HalBuffer
    may get destroyed after the device.
    """
    res = fn(devices)
    gc.collect()
    return res


def get_iree_devices(
    *, driver: str | None = None, device_count: int = 1
) -> List[iree.runtime.HalDevice]:
    """Gets a list of IREE HAL devices for the given driver.

    The first available device_count devices will be created,
    unless the IREE_DEVICE environment variable is set to an
    explicit list of device URIs.

    For example, to select HIP devices 5 and 3:
    ```
    export IREE_DEVICE=hip://5,hip://3
    python ...
    ```
    """
    if "IREE_DEVICE" in os.environ:
        device_uris = [d.strip() for d in os.environ["IREE_DEVICE"].split(",")]
        driver_names = [n.split("://")[0] for n in device_uris]
        if driver is not None:
            if any(driver != driver_name for driver_name in driver_names):
                ValueError(
                    f'Inconsistent IREE driver, expected "{driver}" for all devices f{device_uris}'
                )
        if device_count > len(device_uris):
            raise ValueError(
                "Environment variable IREE_DEVICE provides less devices than requested."
            )
        return [
            iree.runtime.get_driver(driver_names[i]).create_device_by_uri(
                device_uris[i]
            )
            for i in range(device_count)
        ]

    hal_driver = iree.runtime.get_driver(driver)
    available_devices = hal_driver.query_available_devices()
    if driver in ["local-task", "local-sync"]:
        # Use the same actual device for all devices.
        return [
            hal_driver.create_device(available_devices[0]) for _ in range(device_count)
        ]
    else:
        return [
            hal_driver.create_device(available_devices[i]) for i in range(device_count)
        ]


def load_iree_module(
    module_path: str,
    devices: List[iree.runtime.HalDevice],
    parameters_path: Optional[str] = None,
    debug_sink: Optional[iree.runtime.HalModuleDebugSink] = None,
) -> Tuple[iree.runtime.VmModule, iree.runtime.VmContext, iree.runtime.VmInstance]:
    """The VmContext and VmInstance need to outlive the VmModule and any device
    buffers."""
    vm_instance = iree.runtime.VmInstance()
    hal_module = iree.runtime.create_hal_module(
        instance=vm_instance, devices=devices, debug_sink=debug_sink
    )
    modules = [hal_module]
    if parameters_path is not None:
        params_path = Path(parameters_path)
        parameter_index = iree.runtime.ParameterIndex()
        if len(devices) > 1:
            # TODO: make IREE able to load the parameters from the top parameter file
            # without having to specify the parameter file for each shard separately.
            for i in range(len(devices)):
                parameter_index.load(
                    file_path=str(
                        Path(params_path).with_suffix(f".rank{i}{params_path.suffix}")
                    )
                )
        else:
            parameter_index.load(file_path=str(params_path))
        parameter_provider = parameter_index.create_provider(scope="model")
        parameters_module = iree.runtime.create_io_parameters_module(
            vm_instance, parameter_provider
        )
        modules.append(parameters_module)
    vm_module = iree.runtime.VmModule.mmap(vm_instance, str(module_path))
    modules.append(vm_module)
    vm_context = iree.runtime.VmContext(instance=vm_instance, modules=modules)
    return vm_module, vm_context, vm_instance


def promote_bfloat16_to_float32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor.to(dtype=torch.float32)
    else:
        return tensor


def device_array_to_host(device_array: iree.runtime.DeviceArray) -> torch.Tensor:
    def reinterpret_hal_buffer_view_element_type(
        buffer_view: iree.runtime.HalBufferView,
        element_type: iree.runtime.HalElementType,
    ) -> iree.runtime.HalBufferView:
        return iree.runtime.HalBufferView(
            buffer=buffer_view.get_buffer(),
            shape=buffer_view.shape,
            element_type=element_type,
        )

    def reinterpret_device_array_dtype(
        device_array: iree.runtime.DeviceArray, dtype: np.dtype
    ) -> iree.runtime.DeviceArray:
        return iree.runtime.DeviceArray(
            device=device_array._device,
            buffer_view=reinterpret_hal_buffer_view_element_type(
                device_array._buffer_view,
                iree.runtime.array_interop.map_dtype_to_element_type(dtype),
            ),
        )

    # Circumvent the lack of bfloat16 in numpy.
    # TODO: This uses private fields _device and _buffer_view in iree.runtime.DeviceArray.
    # Improve DeviceArray to provide a hatchet to allow for reinterpretation of
    # element type of the underlying buffer.
    def bfloat16_device_array_to_torch(
        device_array: iree.runtime.DeviceArray,
    ) -> torch.Tensor:
        device_array_as_int16 = reinterpret_device_array_dtype(device_array, np.int16)
        torch_tensor_as_int16 = torch.tensor(device_array_as_int16.to_host())
        return torch_tensor_as_int16.view(dtype=torch.bfloat16)

    if device_array._buffer_view.element_type == int(
        iree.runtime.HalElementType.BFLOAT_16
    ):
        return bfloat16_device_array_to_torch(device_array)
    else:
        return torch.tensor(device_array.to_host())


def torch_tensor_to_device_array(
    tensor: torch.Tensor, device: iree.runtime.HalDevice
) -> iree.runtime.DeviceArray:
    if tensor.dtype == torch.bfloat16:
        tensor_as_int16 = tensor.view(dtype=torch.int16)
        device_array_as_int16 = iree.runtime.asdevicearray(
            device, unbox_tensor(tensor_as_int16).to("cpu").detach().numpy()
        )
        buffer_view = iree.runtime.HalBufferView(
            buffer=device_array_as_int16._buffer_view.get_buffer(),
            shape=device_array_as_int16._buffer_view.shape,
            element_type=iree.runtime.HalElementType.BFLOAT_16,
        )
        return iree.runtime.DeviceArray(device, buffer_view)

    return iree.runtime.asdevicearray(
        device, unbox_tensor(tensor).to("cpu").detach().numpy()
    )


def run_iree_module_function(
    module: iree.runtime.VmModule,
    vm_context: iree.runtime.VmContext,
    args: List[iree.runtime.DeviceArray],
    device: iree.runtime.HalDevice,
    function_name: str = "main",
    trace_path_prefix: Optional[str] = None,
) -> List[iree.runtime.DeviceArray]:
    """Run IREE module function with optional tracing of arguments/results."""
    vm_function = module.lookup_function(function_name)
    invoker = iree.runtime.FunctionInvoker(
        vm_context=vm_context,
        # TODO: rework iree.runtime.FunctionInvoker interface for multiple devices.
        # This works, but does not look right.
        device=device,
        vm_function=vm_function,
    )

    if trace_path_prefix is not None:
        for i, arg in enumerate(args):
            np.save(
                f"{trace_path_prefix}{function_name}_arg{i}.npy",
                promote_bfloat16_to_float32(device_array_to_host(arg)).detach().numpy(),
            )
    results = invoker(*args)
    if isinstance(results, iree.runtime.DeviceArray):
        results = (results,)

    if trace_path_prefix is not None:
        for i, arg in enumerate(args):
            np.save(
                f"{trace_path_prefix}{function_name}_arg{i}_post_call.npy",
                device_array_to_host(arg).detach().numpy(),
            )
        for i, arg in enumerate(results):
            np.save(
                f"{trace_path_prefix}{function_name}_result{i}.npy",
                promote_bfloat16_to_float32(device_array_to_host(arg)).detach().numpy(),
            )
    return results


def prepare_iree_module_function_args(
    args: List[Union[AnyTensor, List[AnyTensor]]], devices: List[iree.runtime.HalDevice]
) -> List[iree.runtime.DeviceArray]:
    """Flatten composite tensors into their parts and place them on devices.
    Sharded tensors become a list of their shards while placing them onto their
    corresponding device.
    All unsharded tensors go on device 0.
    """
    res = []
    for arg in args:
        if isinstance(arg, ShardedTensor):
            assert len(devices) == len(arg.shards)
            res.extend(
                [
                    prepare_iree_module_function_args([shard], [device])[0]
                    for shard, device in zip(arg.shards, devices)
                ]
            )
        elif isinstance(arg, (DefaultPrimitiveTensor, torch.Tensor)):
            res.append(torch_tensor_to_device_array(arg, devices[0]))
        else:
            assert isinstance(arg, collections.abc.Sequence)
            res.extend(prepare_iree_module_function_args(arg, devices))
    return res


def flatten_for_iree_signature(tree: Tree) -> List[torch.Tensor]:
    """Flatten a tree of arguments or results for an IREE call.
    E.g. sharded tensors gets flattened into their shards."""
    return torch_tree_flatten(tree)[0]


def call_torch_module_function(
    module: torch.nn.Module,
    function_name: str,
    args: Optional[tuple[AnyTensor]] = None,
    kwargs: Optional[OrderedDict] = None,
    trace_path_prefix: Optional[str] = None,
):
    """Call a torch module function with optional tracing.
    For tracing the arguments/results are flattened to match IREE's signature."""
    args = args if args is not None else tuple()
    kwargs = kwargs if kwargs is not None else OrderedDict()
    assert isinstance(
        kwargs, OrderedDict
    ), "Make sure when flattening the order is preserved"
    if trace_path_prefix is not None:
        flat_args = flatten_for_iree_signature([args, kwargs])
        for i, arg in enumerate(flat_args):
            np.save(
                f"{trace_path_prefix}{function_name}_arg{i}.npy",
                promote_bfloat16_to_float32(arg.to("cpu")).detach().numpy(),
            )
    res = getattr(module, function_name)(*args, **kwargs)
    if trace_path_prefix is not None:
        flat_args = flatten_for_iree_signature([args, kwargs])
        for i, arg in enumerate(flat_args):
            np.save(
                f"{trace_path_prefix}{function_name}_arg{i}.npy",
                promote_bfloat16_to_float32(arg.to("cpu")).detach().numpy(),
            )
        results = (
            (res,)
            if isinstance(
                res,
                (
                    torch.Tensor,
                    InferenceTensor,
                ),
            )
            else res
        )
        flat_results = flatten_for_iree_signature(results)
        for i, result in enumerate(flat_results):
            np.save(
                f"{trace_path_prefix}{function_name}_result{i}.npy",
                result.to("cpu").detach().numpy(),
            )
    return res


def iree_to_torch(*tensors: iree.runtime.DeviceArray) -> List[torch.Tensor]:
    return [device_array_to_host(tensor) for tensor in tensors]


def make_hal_buffer_view_trace_default_callback(
    device: iree.runtime.HalDevice,
) -> iree.runtime.HalModuleBufferViewTraceCallback:
    """Will sink into whatever is configured in the utils.debugging module.

    Ideally we would like to not have to specify the device, but we can't reliably get
    the array on the host from HalBufferView if the memory is not host-mappable.
    In that case a copy from device-to-host needs to be executed."""
    from . import debugging

    class Callback:
        def __init__(self, device: iree.runtime.HalDevice):
            # Make sure we don't create a circular reference.
            self.device = device

        def __call__(self, key: str, buffer_views: List[iree.runtime.HalBufferView]):
            tensors = [
                device_array_to_host(iree.runtime.DeviceArray(self.device, buffer_view))
                for buffer_view in buffer_views
            ]
            debugging.get_trace_tensor_callback()(key, *tensors)

    return Callback(device)
