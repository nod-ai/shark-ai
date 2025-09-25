# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import tempfile
import iree.compiler
from pathlib import Path
from iree.turbine.aot import *
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    prepare_iree_module_function_args,
    run_iree_module_function,
    iree_to_torch,
)

DEFAULT_COMPILE_FLAGS = [
    "--iree-hal-target-device=hip",     # change to your backend (e.g., local, cuda, vulkan)
    "--iree-hip-target=gfx942",         # MI300 example; adjust to your GPU if needed
    "--iree-execution-model=async-external",
    "--iree-opt-strip-assertions=true",
    "--iree-opt-level=O3",
    "--iree-dispatch-creation-propagate-collapse-across-expands=true",
    "--iree-stream-affinity-solver-max-iterations=1024",
    "--iree-hal-indirect-command-buffers=true",
    "--iree-stream-resource-memory-model=discrete",
    "--iree-hip-specialize-dispatches",
    "--iree-hal-memoization=true",
    "--iree-codegen-enable-default-tuning-specs=true"
]


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)

def export_torch_module_to_mlir(
    module: torch.nn.Module,
    args=(),
    kwargs=None,
    *,
    mlir_path: Path,
    target_fn="run_forward",
):
    """
    Export torch module to MLIR and get torch eager reference output.

    Args:
        module: torch.nn.Module under test
        args: example positional inputs (tuple required)
        kwargs: example kwargs
        mlir_path: Path where to save the MLIR file
        target_fn: name of the exported function

    Returns:
        Tuple of (torch_eager_output, export_output)
    """
    kwargs = kwargs or {}
    args = _as_tuple(args)

    # ---- Torch eager reference ----
    module.eval()
    with torch.no_grad():
        expected = module(*args, **kwargs)

    fxb = FxProgramsBuilder(module)
    
    # empty tensors for export input
    # there needs to be one corresponding to each arg
    # NOTE: assuming args are not nested.
    empty_args = tuple([
        torch.empty(arg.shape, dtype=arg.dtype) for arg in args
    ])

    # need to get this info from the test, currently only for static shapes
    # one corresponding to each arg
    dynamic_shapes = tuple([dict() for _ in args])

    print(f"dyn shapes : {dynamic_shapes}")
    print(f"args {args}")

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


def compile_mlir_to_vmfb(
    mlir_path: Path,
    vmfb_path: Path,
    *,
    compile_flags=None,
):
    """
    Compile MLIR file to VMFB.

    Args:
        mlir_path: Path to the MLIR file
        vmfb_path: Path where to save the VMFB file
        compile_flags: List of compilation flags (uses DEFAULT_COMPILE_FLAGS if None)
    """
    compile_flags = compile_flags or DEFAULT_COMPILE_FLAGS

    iree.compiler.compile_file(
        str(mlir_path),
        output_file=str(vmfb_path),
        extra_args=compile_flags,
    )


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
    )

    return iree_out


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
        iree_output: Output from IREE module
        torch_output: Output from torch eager execution
        atol/rtol: tolerances passed to torch.testing.assert_close
    """
    # Convert and compare
    actual = iree_to_torch(*iree_output)
    expected = torch_output

    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    if isinstance(actual, torch.Tensor):
        actual = (actual,)

    # Match dtypes to be safe (IREE may produce f32 by default in some paths)
    actual = tuple(a.to(e.dtype) if hasattr(a, "dtype") else a for a, e in zip(actual, expected))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(f"actual : {actual}")
    print(f"expected : {expected}")


def run_iree_vs_torch_fx(
    module: torch.nn.Module,
    args=(),
    kwargs=None,
    *,
    atol=1e-4,
    rtol=0.0,
    entrypoint="run_forward",
    parameters_path=None,
    compile_flags=None,
    driver="hip",
    device_count=1,
):
    """
    Wrapper for MLIR export via FxProgramsBuilder(model) and IREE vs Torch eager comparison.

    Args:
      module: torch.nn.Module under test
      args: example positional inputs (tuple required)
      kwargs: example kwargs
      atol/rtol: tolerances passed to torch.testing.assert_close
      entrypoint: the method name exported/invoked ("run_forward" by default)
      parameters_path: Optional path to parameters file
      compile_flags: List of compilation flags (uses DEFAULT_COMPILE_FLAGS if None)
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
            args=args,
            kwargs=kwargs,
            mlir_path=mlir_path,
            target_fn=entrypoint,
        )

        # Compile MLIR to VMFB
        compile_mlir_to_vmfb(
            mlir_path=mlir_path,
            vmfb_path=vmfb_path,
            compile_flags=compile_flags,
        )

        # Run with IREE
        iree_output = run_iree_module_from_vmfb(
            vmfb_path=vmfb_path,
            args=args,
            entrypoint=entrypoint,
            parameters_path=parameters_path,
            driver=driver,
            device_count=device_count,
        )

        # Compare outputs
        compare_iree_torch_outputs(
            iree_output=iree_output,
            torch_output=torch_output,
            atol=atol,
            rtol=rtol,
        )
