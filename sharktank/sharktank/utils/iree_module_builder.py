# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""High-level utilities for converting torch modules to IREE modules."""

from typing import Any, Optional, Sequence
from pathlib import Path
import tempfile

import torch
from iree.turbine import aot
from iree.turbine.aot import FxProgramsBuilder
import iree.runtime

from .iree import (
    TorchLikeIreeModule,
    TypePreservingIreeModule,
    get_iree_devices,
    load_iree_module,
    with_iree_device_context,
)


def compile_torch_module_to_iree(
    module: torch.nn.Module,
    example_args: tuple[Any, ...] = tuple(),
    example_kwargs: dict[str, Any] = None,
    function_name: str = "forward",
    compile_args: Sequence[str] = None,
    save_mlir_to: Optional[Path] = None,
    save_vmfb_to: Optional[Path] = None,
) -> memoryview:
    """Compile a torch module to IREE VMFB bytecode.

    Args:
        module: The torch.nn.Module to compile
        example_args: Example positional arguments for tracing
        example_kwargs: Example keyword arguments for tracing
        function_name: Name of the function to export (default: "forward")
        compile_args: Additional IREE compiler flags
        save_mlir_to: Optional path to save the exported MLIR
        save_vmfb_to: Optional path to save the compiled VMFB

    Returns:
        A memoryview of the compiled VMFB bytecode

    Example:
        >>> model = MyTorchModel()
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> vmfb_bytes = compile_torch_module_to_iree(
        ...     model,
        ...     example_args=(example_input,),
        ...     compile_args=["--iree-hal-target-device=local-task"]
        ... )
    """
    if example_kwargs is None:
        example_kwargs = {}

    # Export to MLIR using turbine
    fxb = FxProgramsBuilder(module)

    @fxb.export_program(
        name=function_name, args=example_args, kwargs=example_kwargs, strict=False
    )
    def _(module, *args, **kwargs):
        return getattr(module, function_name)(*args, **kwargs)

    export_output = aot.export(fxb)

    # Save MLIR if requested
    if save_mlir_to is not None:
        export_output.save_mlir(save_mlir_to)

    # Set compiler flags
    if compile_args is not None:
        export_output.session.set_flags(*compile_args)

    # Compile to VMFB
    if save_vmfb_to is not None:
        export_output.compile(save_to=str(save_vmfb_to), target_backends=None)
        with open(save_vmfb_to, "rb") as f:
            return memoryview(f.read())
    else:
        return export_output.compile(save_to=None, target_backends=None).map_memory()


def load_torch_module_as_iree(
    module: torch.nn.Module,
    example_args: tuple[Any, ...] = tuple(),
    example_kwargs: dict[str, Any] = None,
    function_name: str = "forward",
    device: str | list[str] = "local-task",
    device_count: int | None = None,
    compile_args: Sequence[str] = None,
    parameters_path: Optional[str] = None,
    save_mlir_to: Optional[Path] = None,
    save_vmfb_to: Optional[Path] = None,
    output_type_mapper: Optional[Any] = None,
) -> TorchLikeIreeModule:
    """Compile a torch module to IREE and load it as a TorchLikeIreeModule.

    This is a high-level convenience function that combines export, compilation,
    and loading into a single call.

    Args:
        module: The torch.nn.Module to compile
        example_args: Example positional arguments for tracing
        example_kwargs: Example keyword arguments for tracing
        function_name: Name of the function to export (default: "forward")
        device: IREE device(s) to load on (e.g., "local-task", "hip://0")
        device_count: Number of devices to use (for multi-device scenarios)
        compile_args: Additional IREE compiler flags
        parameters_path: Optional path to external parameters (IRPA file)
        save_mlir_to: Optional path to save the exported MLIR
        save_vmfb_to: Optional path to save the compiled VMFB
        output_type_mapper: Optional function to transform IREE's flat tensor tuple
            back to the original output structure. If provided, returns a
            TypePreservingIreeModule (subclass of TorchLikeIreeModule).

    Returns:
        A TorchLikeIreeModule that can be called like the original torch module.
        If output_type_mapper is provided, returns TypePreservingIreeModule.

    Example:
        >>> model = MyTorchModel()
        >>> example_input = torch.randn(1, 3, 224, 224)
        >>> iree_model = load_torch_module_as_iree(
        ...     model,
        ...     example_args=(example_input,),
        ...     device="local-task"
        ... )
        >>> output = iree_model.forward(example_input)
        >>>
        >>> # With type preservation (returns single tensor instead of tuple)
        >>> def unwrap(outputs): return outputs[0]
        >>> iree_model = load_torch_module_as_iree(
        ...     model,
        ...     example_args=(example_input,),
        ...     output_type_mapper=unwrap
        ... )
        >>> output = iree_model.forward(example_input)  # Single tensor, not tuple
    """
    # Compile the module
    vmfb_bytes = compile_torch_module_to_iree(
        module=module,
        example_args=example_args,
        example_kwargs=example_kwargs,
        function_name=function_name,
        compile_args=compile_args,
        save_mlir_to=save_mlir_to,
        save_vmfb_to=save_vmfb_to,
    )

    # Get devices
    iree_devices = get_iree_devices(device=device, device_count=device_count)

    # Load the module
    def load_fn(devices: list[iree.runtime.HalDevice]) -> TorchLikeIreeModule:
        vm_module, vm_context, vm_instance = load_iree_module(
            module_buff=vmfb_bytes,
            devices=devices,
            parameters_path=parameters_path,
            tensor_parallel_size=len(devices),
        )
        if output_type_mapper is not None:
            return TypePreservingIreeModule(
                vm_module, vm_context, devices, output_type_mapper
            )
        else:
            return TorchLikeIreeModule(vm_module, vm_context, devices)

    return with_iree_device_context(load_fn, iree_devices)


def oneshot_compile_and_run(
    module: torch.nn.Module,
    args: tuple[Any, ...] = tuple(),
    kwargs: dict[str, Any] = None,
    function: str = "forward",
    device: str | list[str] = "local-task",
    device_count: int | None = None,
    compile_args: Sequence[str] = None,
) -> tuple[torch.Tensor, ...]:
    """One-shot function: export, compile, load, and run in one call.

    This is useful for quick testing and benchmarking. For production use,
    prefer load_torch_module_as_iree() to reuse the compiled module.

    Args:
        module: The torch.nn.Module to run
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        function: Name of the function to call (default: "forward")
        device: IREE device(s) to run on
        device_count: Number of devices to use
        compile_args: Additional IREE compiler flags

    Returns:
        Tuple of output tensors

    Example:
        >>> model = MyTorchModel()
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> outputs = oneshot_compile_and_run(
        ...     model,
        ...     args=(input_tensor,),
        ...     device="local-task"
        ... )
    """
    if kwargs is None:
        kwargs = {}

    iree_module = load_torch_module_as_iree(
        module=module,
        example_args=args,
        example_kwargs=kwargs,
        function_name=function,
        device=device,
        device_count=device_count,
        compile_args=compile_args,
    )

    return getattr(iree_module, function)(*args, **kwargs)
