# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import tempfile
import iree.compiler
from pathlib import Path
from iree.turbine.aot import FxProgramsBuilder, export
from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    prepare_iree_module_function_args,
    run_iree_module_function,
    iree_to_torch,
)


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

    fxb = FxProgramsBuilder(module)
    
    # empty tensors for export input
    # there needs to be one corresponding to each arg
    # NOTE: assuming args are not nested.
    empty_args = tuple([
        torch.empty(arg.shape, dtype=arg.dtype) for arg in input_args
    ])

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


def compile_mlir_to_vmfb(
    mlir_path: Path,
    vmfb_path: Path,
    *,
    compile_flags: list[str],
):
    """
    Compile MLIR file to VMFB.

    Args:
        mlir_path: Path to the MLIR file
        vmfb_path: Path where to save the VMFB file
        compile_flags: List of compilation flags for iree
    """

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
    compile_flags: list[str]|None=None,
    driver="hip",
    device_count=1,
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

        compile_mlir_to_vmfb(
            mlir_path=mlir_path,
            vmfb_path=vmfb_path,
            compile_flags=compile_flags,
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

        # Compare outputs
        compare_iree_torch_outputs(
            iree_output=iree_output,
            torch_output=torch_output,
            atol=atol,
            rtol=rtol,
        )
