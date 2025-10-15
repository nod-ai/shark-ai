# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, List, Tuple, Optional, Union, overload, TYPE_CHECKING
import os
import sys
import json
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import collections.abc
from collections import OrderedDict
from contextlib import contextmanager
import subprocess
import gc
import torch

import iree.compiler
from sharktank.types.tensors import (
    AnyTensor,
    InferenceTensor,
    ShardedTensor,
    DefaultPrimitiveTensor,
    unbox_tensor,
    torch_tree_flatten,
)
from sharktank.utils import verify_exactly_one_is_not_none
from .tree import Tree
from iree.runtime import FileHandle
import iree.runtime

from iree.turbine import aot
from iree.turbine.aot import export


if TYPE_CHECKING:
    from ..layers import ModelConfig

torch_dtype_to_hal_element_type_map = {
    torch.float8_e4m3fnuz: iree.runtime.HalElementType.FLOAT_8_E4M3_FNUZ,
    torch.float8_e4m3fn: iree.runtime.HalElementType.FLOAT_8_E4M3_FN,
    torch.bfloat16: iree.runtime.HalElementType.BFLOAT_16,
}

hal_element_type_to_torch_dtype_map = {
    v: k for k, v in torch_dtype_to_hal_element_type_map.items()
}

dtype_to_dtype_reinterpret_map = {
    torch.float8_e4m3fnuz: torch.int8,
    torch.float8_e4m3fn: torch.int8,
    torch.bfloat16: torch.int16,
}
"""We want to map dtypes unsupported by iree.runtime.DeviceArray.
This is due to numpy having no support for these and we need reinterpretation
of the data in order to get it across the torch-IREE bundary.
"""


torch_dtype_to_numpy_dtype_map = {
    torch.int8: np.int8,
    torch.int16: np.int16,
}


def oneshot_iree_run(
    module: torch.nn.Module,
    args: tuple[Any, ...] = tuple(),
    kwargs: dict[str, Any] = {},
    function: str = "forward",
    device: str | list[str] = "local-task",
    device_count: int | None = None,
    compile_args: tuple[str, ...] = None,
) -> tuple[torch.Tensor, ...]:
    """All in one: export, compile and run."""
    from iree.turbine import aot
    from iree.turbine.aot import FxProgramsBuilder

    fxb = FxProgramsBuilder(module)

    @fxb.export_program(name=function, args=args, kwargs=kwargs, strict=False)
    def _(module, *args, **kwargs):
        return getattr(module, function)(*args, **kwargs)

    export_output = aot.export(
        fxb,
    )
    if compile_args is not None:
        export_output.session.set_flags(*compile_args)
    memory_view: memoryview = export_output.compile(
        save_to=None, target_backends=None
    ).map_memory()
    iree_devices = get_iree_devices(device=device, device_count=device_count)

    def run(iree_devices: list[iree.runtime.HalDevice]):
        vm_module, vm_context, vm_instance = load_iree_module(
            module_buff=memory_view, devices=iree_devices
        )
        torch_like_iree_module = TorchLikeIreeModule(
            vm_module, vm_context, iree_devices
        )
        results = getattr(torch_like_iree_module, function)(*args, **kwargs)
        # Clone to avoid leaking IREE-backed torch tensors.
        results = tuple(t.clone() for t in results)
        return results

    return with_iree_device_context(run, iree_devices)


class TorchLikeIreeModule:
    """Makes an IREE module look like a torch module. Where it can be called with
    Sharktank and Torch tensors.

    This handles marshaling of torch tensor and sharktank.type.InferenceTensor arguments.
    Unfortunately, we can't marshall the output back to the correct tensor types as
    some of the information is lost. E.g. the sharded tensor types. We return a flat
    list of torch tensors.
    """

    def __init__(
        self,
        module: iree.runtime.VmModule,
        vm_context: iree.runtime.VmContext,
        devices: List[iree.runtime.HalDevice],
    ):
        self.module = module
        self.vm_context = vm_context
        self.devices = devices

    def __getattr__(self, name: str) -> Any:
        def f(
            *args: tuple[Any, ...], **kwargs: dict[str, Any]
        ) -> tuple[torch.Tensor, ...]:
            flat_args = flatten_for_iree_signature(
                (
                    args,
                    kwargs,
                )
            )
            iree_args = prepare_iree_module_function_args(flat_args, self.devices)
            res = run_iree_module_function(
                module=self.module,
                vm_context=self.vm_context,
                args=iree_args,
                device=self.devices[0],
                function_name=name,
            )
            res = iree_to_torch(*res)

            # Copy back to args as they may have been modified in-place by the function.
            iree_args_post_call = iree_to_torch(*iree_args)
            for arg, iree_arg in zip(flat_args, iree_args_post_call):
                arg[...] = iree_arg

            return res

        return f


def get_file_handle(
    weight_path: Path,
    shard_count: int = 1,
) -> List[FileHandle]:
    handles = []
    modified_weight_path = weight_path
    for i in range(shard_count):
        if shard_count > 1:
            modified_weight_path = str(
                weight_path.with_suffix(f".rank{i}{weight_path.suffix}")
            )
        with open(str(modified_weight_path), "rb") as f:
            handles.append(FileHandle.wrap_fd(f.fileno()))
    return handles


def get_iree_compiler_flags(
    iree_hal_target_device: str,
    iree_hal_local_target_device_backends: list[str] | None = None,
    iree_hip_target: str | None = None,
    device_count: int = 1,
) -> list[str]:
    """Retrieve compiler flags driven by the test configuration."""
    res = []
    if device_count == 1:
        res += [f"--iree-hal-target-device={iree_hal_target_device}"]
    else:
        res += [
            f"--iree-hal-target-device={iree_hal_target_device}[{i}]"
            for i in range(device_count)
        ]

    if iree_hal_target_device.startswith("local"):
        res += [
            f"--iree-hal-local-target-device-backends={v}"
            for v in iree_hal_local_target_device_backends
        ]
        res += ["--iree-llvmcpu-target-cpu=host"]
    elif iree_hal_target_device.startswith("hip"):
        res += [f"--iree-hip-target={iree_hip_target}"]
    else:
        raise ValueError(
            f'"{iree_hal_target_device}" is not a supported IREE HAL target device'
        )

    return res


def get_iree_compiler_flags_from_object(o: Any, device_count: int = 1) -> list[str]:
    kwargs = {
        "iree_hal_target_device": o.iree_hal_target_device,
        "device_count": device_count,
    }
    if hasattr(o, "iree_hal_local_target_device_backends"):
        kwargs[
            "iree_hal_local_target_device_backends"
        ] = o.iree_hal_local_target_device_backends
    if hasattr(o, "iree_hip_target"):
        kwargs["iree_hip_target"] = o.iree_hip_target

    return get_iree_compiler_flags(**kwargs)


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


@overload
def get_iree_devices(
    *, driver: str, device_count: int, allow_repeat: bool
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
    ...


@overload
def get_iree_devices(
    *, device: str | list[str], device_count: int, allow_repeating: bool
) -> List[iree.runtime.HalDevice]:
    ...


def get_iree_devices(
    *,
    device: str | list[str] | None = None,
    driver: str | None = None,
    device_count: int | None = None,
    allow_repeating: bool = True,
) -> List[iree.runtime.HalDevice]:
    has_device_arg = device is not None
    has_driver_arg = driver is not None
    has_device_count_arg = device_count is not None
    if not (has_device_arg or has_driver_arg or has_device_count_arg):
        raise ValueError(
            "Could not select overload. Please, provide at least one argument"
        )
    if has_device_arg and has_driver_arg:
        raise ValueError(
            "device arg is mutually exclusive with driver and device_count args"
        )
    if has_driver_arg and not has_device_count_arg:
        raise ValueError("When driver is provided, device_count must also be provided")

    if has_device_arg:
        if isinstance(device, str):
            device = [device]
    elif "IREE_DEVICE" in os.environ:
        device_uris = [d.strip() for d in os.environ["IREE_DEVICE"].split(",")]
        driver_names = [n.split("://")[0] for n in device_uris]
        if driver is not None:
            if any(driver != driver_name for driver_name in driver_names):
                ValueError(
                    f'Inconsistent IREE driver, expected "{driver}" for all devices f{device_uris}'
                )
        device = device_uris

    if device is not None:
        if not has_device_count_arg:
            device_count = len(device)
        if device_count < len(device):
            device = device[:device_count]
        hal_devices = [iree.runtime.get_device(d, cache=False) for d in device]
    else:
        hal_driver = iree.runtime.get_driver(driver)
        device_infos = hal_driver.query_available_devices()
        if device_count < len(device_infos):
            device_infos = device_infos[:device_count]
        hal_devices = [
            hal_driver.create_device(device_info) for device_info in device_infos
        ]

    if not allow_repeating and len(hal_devices) < device_count:
        ValueError("Requested more devices than available or specified")

    return [hal_devices[i % len(hal_devices)] for i in range(device_count)]


_same_as_device_count = object()


def load_iree_module(
    *,
    module_buff: bytearray | None = None,
    module_path: str | None = None,
    devices: List[iree.runtime.HalDevice],
    parameters_path: Optional[str] = None,
    debug_sink: Optional[iree.runtime.HalModuleDebugSink] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> Tuple[iree.runtime.VmModule, iree.runtime.VmContext, iree.runtime.VmInstance]:
    """The VmContext and VmInstance need to outlive the VmModule and any device
    buffers."""
    verify_exactly_one_is_not_none(module_buff=module_buff, module_path=module_path)

    parallel_size = tensor_parallel_size * pipeline_parallel_size
    assert parallel_size == len(devices)

    vm_instance = iree.runtime.VmInstance()
    hal_module = iree.runtime.create_hal_module(
        instance=vm_instance, devices=devices, debug_sink=debug_sink
    )
    modules = [hal_module]
    if parameters_path is not None:
        params_path = Path(parameters_path)
        parameter_index = iree.runtime.ParameterIndex()
        if parallel_size == len(devices):
            # TODO: make IREE able to load the parameters from the top parameter file
            # without having to specify the parameter file for each shard separately.
            handles = get_file_handle(
                shard_count=tensor_parallel_size, weight_path=params_path
            )
            for handle in handles:
                parameter_index.load_from_file_handle(handle, "irpa")
        else:
            raise NotImplementedError(
                f"pipeline_parallelism_size * tensor_parallelism_size {parallel_size} != len(devices) {len(devices)}"
            )
        parameter_provider = parameter_index.create_provider(scope="model")
        parameters_module = iree.runtime.create_io_parameters_module(
            vm_instance, parameter_provider
        )
        modules.append(parameters_module)
    if module_path is not None:
        vm_module = iree.runtime.VmModule.mmap(vm_instance, module_path)
    else:
        vm_module = iree.runtime.VmModule.copy_buffer(vm_instance, module_buff)
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
            element_type=int(element_type),
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

    # Circumvent the lack of bfloat16, float8_e4m3fnuz, etc. in numpy.
    # TODO: This uses private fields _device and _buffer_view in iree.runtime.DeviceArray.
    # Improve DeviceArray to provide a hatchet to allow for reinterpretation of
    # element type of the underlying buffer.
    def device_array_to_torch_via_reinterpret(
        device_array: iree.runtime.DeviceArray,
    ) -> torch.Tensor:
        hal_element_type = iree.runtime.HalElementType(
            device_array._buffer_view.element_type
        )
        reinterpret_torch_dtype: torch.dtype = dtype_to_dtype_reinterpret_map[
            hal_element_type_to_torch_dtype_map[hal_element_type]
        ]
        reinterpret_numpy_dtype: np.dtype = torch_dtype_to_numpy_dtype_map[
            reinterpret_torch_dtype
        ]
        device_array_reinterpreted_dtype = reinterpret_device_array_dtype(
            device_array, reinterpret_numpy_dtype
        )
        torch_tensor_reinterpreted_dtype = torch.tensor(
            device_array_reinterpreted_dtype.to_host()
        )
        return torch_tensor_reinterpreted_dtype.view(
            dtype=hal_element_type_to_torch_dtype_map[hal_element_type]
        )

    hal_element_type = iree.runtime.HalElementType(
        device_array._buffer_view.element_type
    )
    if (
        hal_element_type in hal_element_type_to_torch_dtype_map
        and hal_element_type_to_torch_dtype_map[hal_element_type]
        in dtype_to_dtype_reinterpret_map
    ):
        return device_array_to_torch_via_reinterpret(device_array)
    else:
        return torch.tensor(device_array.to_host())


def tensor_to_device_array(
    tensor: torch.Tensor | DefaultPrimitiveTensor, device: iree.runtime.HalDevice
) -> iree.runtime.DeviceArray:
    if tensor.dtype in torch_dtype_to_hal_element_type_map.keys():
        tensor_reinterpreted_dtype = unbox_tensor(tensor).view(
            dtype=dtype_to_dtype_reinterpret_map[tensor.dtype]
        )
        device_array_reinterpreted_dtype = iree.runtime.asdevicearray(
            device, unbox_tensor(tensor_reinterpreted_dtype).to("cpu").detach().numpy()
        )
        buffer_view = iree.runtime.HalBufferView(
            buffer=device_array_reinterpreted_dtype._buffer_view.get_buffer(),
            shape=device_array_reinterpreted_dtype._buffer_view.shape,
            element_type=torch_dtype_to_hal_element_type_map[tensor.dtype],
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
    return_type: str = "iree",
) -> List[iree.runtime.DeviceArray | torch.Tensor | np.ndarray]:
    """Run IREE module function with optional tracing of arguments/results.

    return_type: ["iree", "torch", "numpy"] -> it's safer to do this here where we can ensure a good destruction order.
    """
    vm_function = module.lookup_function(function_name)
    if vm_function is None:
        available_functions = module.function_names
        raise ValueError(
            f"Function '{function_name}' not found in module. Available functions: {available_functions}"
        )
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
    if return_type == "iree":
        res = results
    elif return_type == "torch":
        res = iree_to_torch(*results)
    elif return_type == "numpy":
        res = [device_array_to_host(r).detach().numpy() for r in results]
    gc.collect()
    return res


def prepare_iree_module_function_args(
    args: list[Union[AnyTensor, iree.runtime.DeviceArray, list]],
    devices: list[iree.runtime.HalDevice],
) -> list[iree.runtime.DeviceArray]:
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
            res.append(tensor_to_device_array(arg, devices[0]))
        elif isinstance(arg, iree.runtime.DeviceArray):
            res.append(arg)
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
    res = [device_array_to_host(tensor) for tensor in tensors]
    return res


import weakref


def list_stale_objects():
    """
    Identifies and lists stale (uncollectable) objects using the gc module.

    This function forces a garbage collection cycle and then inspects the
    `gc.garbage` list, which contains objects that the collector found to be
    unreachable but could not be freed. It then prints the details of these
    objects, including their type.
    """
    print("Running garbage collection to find stale objects...")
    gc.set_debug(gc.DEBUG_SAVEALL)

    # The gc.collect() function returns the number of unreachable objects found.
    found_objects = gc.collect()
    print(f"Garbage collector found {found_objects} unreachable objects.")

    if not gc.garbage:
        print(
            "\nNo stale (uncollectable) objects found. The 'gc.garbage' list is empty."
        )
        return

    print(f"\n--- Found {len(gc.garbage)} Stale Objects ---")
    for i, obj in enumerate(gc.garbage):
        try:
            # We use repr() to get a developer-friendly string representation of the object.
            obj_repr = repr(obj)
            obj_type = type(obj)
            if any(a in str(obj_type).lower() for a in ["iree", "hal", "buff"]):
                print(f"{i+1}. Object: {obj_repr}")
                print(f"   Type: {obj_type}")
                # weakref.getweakrefcount tells us how many weak references point to the object.
                print(f"   Weak References: {weakref.getweakrefcount(obj)}")
                print("-" * 20)
        except Exception as e:
            print(f"Could not inspect object {i+1} due to an error: {e}")
    print("--- End of List ---")


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


def trace_with_tracy(
    fn: Callable[[int], Any],
    /,
    *,
    output_trace_path: str = None,
    port: int = None,
    capture_extra_args: list[str] | None = None,
) -> Any:
    """Trace a callable with iree-tracy-capture.
    The capture process is started before executing the tracing target.and is waited on
    to finish. The traced target function is started in parallel.
    If a port is not provided a free one is selected automatically.
    """
    capture_cmd = ["iree-tracy-capture", "-f"]
    if output_trace_path:
        capture_cmd += ["-o", output_trace_path]
    if port is None:
        from .io import find_free_port

        port = find_free_port()
    if capture_extra_args:
        capture_cmd += capture_extra_args
    capture_cmd += ["-p", f"{port}"]
    with subprocess.Popen(capture_cmd) as capture_proc:
        try:
            res = fn(port)
        except:
            capture_proc.terminate()
            raise
        capture_process_return_code = capture_proc.wait()
        if capture_process_return_code != 0:
            raise subprocess.CalledProcessError(
                f"Tracy capture process {capture_cmd} failed with return code {capture_process_return_code}"
            )
        return res


def trace_command_with_tracy(
    cmd: list[str],
    /,
    *,
    output_trace_path: str = None,
    port: int = None,
    capture_extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    **run_kwargs,
):
    """Trace an executable with Tracy."""

    def fn(port: int):
        env2 = env or os.environ
        env2 = deepcopy(env2)
        env2["TRACY_PORT"] = str(port)
        proc = subprocess.run(cmd, env=env2, **run_kwargs)
        proc.check_returncode()

    trace_with_tracy(
        fn,
        output_trace_path=output_trace_path,
        port=port,
        capture_extra_args=capture_extra_args,
    )


def trace_model_with_tracy(
    config: "ModelConfig", function: str, output_trace_path: str = None, **kwargs
):
    """Trace an already exported and compiled model with Tracy."""
    cmd = [
        sys.executable,
        "-m",
        "sharktank.tools.trace_model_with_tracy",
        f"--function={function}",
    ]
    if output_trace_path is None:
        output_trace_path = f"{config.iree_module_path}.tracy"
    trace_command_with_tracy(
        cmd,
        input=json.dumps(config.asdict_for_saving()).encode(),
        output_trace_path=output_trace_path,
        **kwargs,
    )


def run_model_with_iree_run_module(
    config: "ModelConfig", function: str, **subprocess_run_kwargs
):
    """Run an already exported and compiled model with iree-run-module.
    It is required that is exports its input arguments.
    """
    cmd = [
        "iree-run-module",
        f"--module={config.iree_module_path}",
        f"--device={config.iree_hal_driver}",
        f"--function={function}",
    ]

    parameters_path = config.export_parameters_path
    if parameters_path is None:
        parameters_path = config.parameters_path
    if parameters_path is not None:
        cmd.append(f"--parameters=model={parameters_path}")

    input_args_descriptor_path = f"{config.mlir_path.stem}-{function}-arg-desc"
    with open(input_args_descriptor_path, "r") as f:
        input_args = f.readlines()
    input_args = [f"--input={arg.strip()}" for arg in input_args]
    cmd += input_args
    subprocess.check_call(cmd, **subprocess_run_kwargs)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def export_torch_module_to_mlir(
    module: torch.nn.Module,
    input_args=(),
    kwargs=None,
    *,
    mlir_path: Path,
    target_fn="run_forward",
):
    """
    Export torch module to MLIR and get torch eager reference output.

    Args:
        module: torch.nn.Module under test
        input_args: example positional inputs (tuple required)
        kwargs: example kwargs
        mlir_path: Path where to save the MLIR file
        target_fn: name of the exported function

    Returns:
        Tuple of (torch_eager_output, export_output)
    """
    kwargs = kwargs or {}
    input_args = _as_tuple(input_args)

    # ---- Torch eager reference ----
    module.eval()
    with torch.no_grad():
        expected = module(*input_args, **kwargs)

    from iree.turbine import aot
    from iree.turbine.aot import FxProgramsBuilder

    fxb = FxProgramsBuilder(module)

    # empty tensors for export input
    # there needs to be one corresponding to each arg
    # NOTE: assuming args are not nested.
    empty_args = tuple([torch.empty(arg.shape, dtype=arg.dtype) for arg in input_args])

    # need to get this info from the test, currently only for static shapes
    # one corresponding to each arg
    dynamic_shapes = tuple([dict() for _ in input_args])

    @fxb.export_program(
        name=target_fn,
        args=empty_args,
        dynamic_shapes=(dynamic_shapes,),
        strict=False,
    )
    def _(module, *fn_args):
        return module.forward(*fn_args)

    export_output = export(fxb, import_symbolic_shape_expressions=True)
    export_output.save_mlir(mlir_path)

    return expected, export_output


def run_iree_module_from_vmfb(
    vmfb_path: Path,
    args=(),
    *,
    entrypoint="run_forward",
    parameters_path=None,
    driver="hip",
    device_count=1,
):
    """
    Load VMFB and run with IREE.

    Args:
        vmfb_path: Path to the VMFB file
        args: Input arguments for the module
        entrypoint: Name of the function to run
        parameters_path: Optional path to parameters file
        driver: IREE driver to use
        device_count: Number of devices

    Returns:
        IREE module output
    """
    args = _as_tuple(args)

    # Load & run with IREE
    devices = get_iree_devices(driver=driver, device_count=device_count)

    def run_with_devices(devices):
        iree_module, vm_context, _ = load_iree_module(
            module_path=str(vmfb_path),
            devices=devices,
            parameters_path=parameters_path,
        )
        iree_args = prepare_iree_module_function_args(args=args, devices=devices)

        iree_out = run_iree_module_function(
            module=iree_module,
            vm_context=vm_context,
            args=iree_args,
            device=devices[0],
            function_name=entrypoint,
            return_type="iree",
        )
        results = iree_to_torch(*iree_out)
        del iree_out
        results = tuple(t.clone() for t in results)

        return results

    return with_iree_device_context(run_with_devices, devices)


def compare_iree_torch_outputs(
    iree_output,
    torch_output,
    *,
    atol=1e-4,
    rtol=0.0,
):
    """
    Compare IREE output with torch eager reference and assert closeness.

    Args:
        iree_output: Output from IREE module as a tuple of np.ndarrays.
        torch_output: Output from torch eager execution
        atol/rtol: tolerances passed to torch.testing.assert_close
    """
    # Convert and compare
    expected = torch_output
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    actual = ()
    for idx, o in enumerate(iree_output):
        if isinstance(o, np.ndarray):
            output = torch.tensor(o).type_as(expected[idx])
        else:
            output = o
        actual += (output,)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def validate_and_get_irpa_path(request):
    """
    Validate and get IRPA path from pytest request configuration.

    Args:
        request: pytest request fixture

    Returns:
        str: Path to the IRPA file

    Raises:
        pytest.skip: If IRPA path is not provided or file doesn't exist
    """
    from pytest import skip

    # Get IRPA path from command line argument
    irpa_path = request.config.getoption("--parameters")

    # Skip test if no IRPA path provided
    if irpa_path is None:
        skip("No IRPA path provided. Use --parameters to specify the IRPA file.")

    # Skip test if IRPA file doesn't exist
    if not Path(irpa_path).exists():
        skip(f"IRPA file not found: {irpa_path}")

    return irpa_path


def run_iree_vs_torch_fx(
    module: torch.nn.Module,
    input_args=(),
    kwargs=None,
    *,
    atol=1e-4,
    rtol=0.0,
    entrypoint="run_forward",
    parameters_path=None,
    compile_flags: list[str] | None = None,
    driver="hip",
    device_count=1,
    directory=".",
):
    """
    Wrapper for MLIR export via FxProgramsBuilder(model) and IREE vs Torch eager comparison.

    Args:
      module: torch.nn.Module under test
      input_args: example positional inputs (tuple required)
      kwargs: example kwargs
      atol/rtol: tolerances passed to torch.testing.assert_close
      entrypoint: the method name exported/invoked ("run_forward" by default)
      parameters_path: Optional path to parameters file
      compile_flags: List of compilation flags for iree
      driver: IREE driver to use
      device_count: Number of devices
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mlir_path = td / "module.mlir"
        vmfb_path = td / "module.vmfb"

        # Export to MLIR and get torch reference
        torch_output, _ = export_torch_module_to_mlir(
            module=module,
            input_args=input_args,
            kwargs=kwargs,
            mlir_path=mlir_path,
            target_fn=entrypoint,
        )

        # Compile MLIR to VMFB
        if compile_flags is None:
            raise ValueError("compile_flags must be provided")

        iree.compiler.compile_file(
            str(mlir_path),
            output_file=str(vmfb_path),
            extra_args=compile_flags,
        )

        # Run with IREE
        iree_output = run_iree_module_from_vmfb(
            vmfb_path=vmfb_path,
            args=input_args,
            entrypoint=entrypoint,
            parameters_path=parameters_path,
            driver=driver,
            device_count=device_count,
        )
        gc.collect()
        # Compare outputs
        compare_iree_torch_outputs(
            iree_output=iree_output,
            torch_output=torch_output,
            atol=atol,
            rtol=rtol,
        )
