
# sharktank/tests/layers/_helpers.py
import torch
import tempfile
from pathlib import Path
import iree.compiler


from sharktank.utils.iree import (
    get_iree_devices,
    load_iree_module,
    prepare_iree_module_function_args,
    run_iree_module_function,
    iree_to_torch,
)

from iree.turbine.aot import *

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

def run_iree_vs_torch_fx(
    module: torch.nn.Module,
    args=(),
    kwargs=None,
    *,
    atol=1e-4,
    rtol=0.0,
    entrypoint="forward",
    parameters_path=None,
):
    """
    Exports MLIR via FxProgramsBuilder(model) and compares IREE vs Torch eager.

    Args:
      module: torch.nn.Module under test
      args: example positional inputs (tuple required)
      kwargs: example kwargs
      atol/rtol: tolerances passed to torch.testing.assert_close
      entrypoint: the method name exported/invoked ("forward" by default)
    """
    kwargs = kwargs or {}
    args = _as_tuple(args)
    torch.manual_seed(1234)
    target_fn = "run_forward"
    entrypoint = target_fn

    # ---- 1) Torch eager reference ----
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

    # Export the selected entry point (callable) from the instance `module`.
    # We pass a bound method so export() can trace that entry.
    # target_fn = getattr(type(module), entrypoint)

    export_output = export(fxb, import_symbolic_shape_expressions=True)

    # The turbine builder attaches a Torch-MLIR operation on the exported program.
    # Retrieve MLIR text and compile it with iree-compile.
    # Note: sharktank's exporter uses the same fx-builder object to drive MLIR generation.
    #   See export_paged_llm_v1.py (fxb usage).
    # mlir_text = ep.mlir_module_operation.get_asm(enable_debug_info=False)

    # Compile MLIR -> VMFB
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mlir_path = td / "module.mlir"
        vmfb_path = td / "module.vmfb"
        export_output.save_mlir(mlir_path)

        iree.compiler.compile_file(
            str(mlir_path),
            output_file=str(vmfb_path),
            extra_args=DEFAULT_COMPILE_FLAGS,
        )

        # Load & run with IREE
        devices = get_iree_devices(driver="hip", device_count=1)  # adjust driver
        iree_module, vm_context, _ = load_iree_module(
            module_path=str(vmfb_path),
            devices=devices,
            parameters_path=parameters_path,
        )
        iree_args = prepare_iree_module_function_args(args=args, devices=devices)

        # For FxProgramsBuilder export, the function name is typically "forward".
        # If you exported a different method, pass entrypoint=<that name>.
        # do we need logic to identify the correct entrypoints, will we have multi entry point executions in these pytests?

        iree_out = run_iree_module_function(
            module=iree_module,
            vm_context=vm_context,
            args=iree_args,
            device=devices[0],
            function_name=entrypoint,
        )

    # TODO: refactor to separate it from iree compile and run
    # Convert and compare
    actual = iree_to_torch(*iree_out)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    if isinstance(actual, torch.Tensor):
        actual = (actual,)

    # Match dtypes to be safe (IREE may produce f32 by default in some paths)
    actual = tuple(a.to(e.dtype) if hasattr(a, "dtype") else a for a, e in zip(actual, expected))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(f"actual : {actual}")
    print(f"expected : {expected}")
