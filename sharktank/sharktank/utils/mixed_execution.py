# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utilities for mixed eager/compiled execution using PyTorch FX graph partitioning.

This module enables selective execution of different parts of a model in eager PyTorch
vs compiled IREE by partitioning the computation graph. Compiled portions are optimized
via IREE while eager portions run in standard PyTorch.

Example usage:
    ```python
    import torch
    from torch.fx import symbolic_trace

    # Define what should run in eager mode
    def should_run_eager(module):
        return module.foo == module.bar

    # Trace and partition model
    traced = symbolic_trace(model)
    partitioned = partition_with_transitions(traced, model, should_run_eager)

    # Compile the compiled partitions and replace with IREE invokers
    compile_partitions(partitioned, example_inputs, iree_devices)

    # Execute with mixed eager/compiled
    result = partitioned(input_tensor)
    ```
"""

from typing import Callable, List, Tuple, Dict, Any, Optional
import torch
from torch.fx import GraphModule, Tracer
from torch.fx.passes.split_module import split_module


# Marker attribute for modules that should run in eager mode
_EAGER_MODE_ATTR = "_run_in_eager_mode"


def eager_mode(cls):
    """
    Decorator to mark a module class as requiring eager execution.
    This is only respected by the mixed execution framework when the default_should_run_eager function is passed.

    Example:
        ```python
        @eager_mode
        class MySpecialLayer(torch.nn.Module):
            def forward(self, x):
                # This will run in eager PyTorch, not compiled IREE
                return x * 2
        ```
    """
    setattr(cls, _EAGER_MODE_ATTR, True)
    return cls


def default_should_run_eager(module: torch.nn.Module) -> bool:
    """
    Default predicate function that checks if a module is marked with @eager_mode.

    Args:
        module: The module instance to check

    Returns:
        True if the module's class is marked with @eager_mode, False otherwise
    """
    return getattr(type(module), _EAGER_MODE_ATTR, False)


def trace_module(
    module: torch.nn.Module,
    should_run_eager_fn: Optional[Callable[[torch.nn.Module], bool]] = None
) -> GraphModule:
    """
    Traces a module using a custom tracer that respects the should_run_eager predicate.

    The should_run_eager function is used to determine which modules should be treated
    as leaf modules (not traced into). This is important because modules that will run
    in eager mode should not be traced - they should remain as atomic call_module operations.

    Args:
        module: The PyTorch module to trace
        should_run_eager_fn: Optional predicate function(module) -> bool.
            If None, uses default_should_run_eager which checks for @eager_mode decorator.

    Returns:
        A traced GraphModule

    Example:
        ```python
        @eager_mode
        class EagerLayer(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.eager = EagerLayer()

            def forward(self, x):
                x = self.linear(x)
                x = self.eager(x)
                return x

        model = MyModel()
        traced = trace_module(model)  # Uses default_should_run_eager
        ```
    """
    if should_run_eager_fn is None:
        should_run_eager_fn = default_should_run_eager

    class CustomTracer(Tracer):
        def __init__(self, should_run_eager_fn):
            super().__init__()
            self.should_run_eager_fn = should_run_eager_fn
        def is_leaf_module(self, module: torch.nn.Module, module_qualified_name: str) -> bool:
            if should_run_eager_fn(module):
                return True
            return super().is_leaf_module(module, module_qualified_name)

    tracer = CustomTracer(should_run_eager_fn)
    graph = tracer.trace(module)
    return GraphModule(module, graph)


def partition_by_predicate(
    traced_graph: GraphModule,
    module: torch.nn.Module,
    should_run_eager_fn: Callable[[torch.nn.Module], bool],
) -> GraphModule:
    """
    Partitions a traced FX graph by creating separate modules for eager and compiled blocks.

    This function automatically generates unique partition names (eager_0, compiled_0, eager_1,
    compiled_1, ...).

    Args:
        traced_graph: The traced FX GraphModule to partition
        module: The original PyTorch module
        should_run_eager_fn: Predicate function that takes a module and
            returns True if the module should run in eager mode, False for compiled

    Returns:
        A partitioned GraphModule where:
        - The top-level forward() orchestrates calls to submodules
        - Submodules are named 'submod_eager_N' or 'submod_compiled_N' where N is the partition index
        - Each submodule is a GraphModule containing a subgraph of the original graph
        - Original modules (nn.Linear, etc.) are preserved within partition submodules

    Example:
        ```python
        def should_run_eager(module):
            return isinstance(module, EagerLayer)

        partitioned = partition_by_predicate(traced, model, should_run_eager)
        print(partitioned.code)
        # def forward(self, x):
        #     submod_compiled_0 = self.submod_compiled_0(x)
        #     submod_eager_1 = self.submod_eager_1(submod_compiled_0)
        #     submod_compiled_2 = self.submod_compiled_2(submod_eager_1)
        #     return submod_compiled_2
        ```
    """
    partition_counter = 0
    current_is_eager = None

    def partition_fn(node):
        nonlocal partition_counter, current_is_eager
        """Internal partition function that assigns unique partition names at transitions."""
        node_is_eager = False
        if node.op == "call_module":
            submod = traced_graph.get_submodule(node.target)
            node_is_eager = should_run_eager_fn(submod)

        if current_is_eager is None:
            current_is_eager = node_is_eager
        elif current_is_eager != node_is_eager:
            partition_counter += 1
            current_is_eager = node_is_eager

        mode = "eager" if current_is_eager else "compiled"
        return f"{mode}_{partition_counter}"

    return split_module(traced_graph, module, partition_fn)


def get_example_inputs_for_partitions(
    partitioned: GraphModule,
    full_model_inputs: Tuple[Any, ...],
) -> Dict[str, Tuple[Any, ...]]:
    """
    Captures example inputs for each partition by tracing execution with hooks.

    This is necessary because IREE compilation needs example inputs for each partition,
    but we only have example inputs for the full model. This function runs the partitioned
    graph once to capture intermediate tensor shapes at partition boundaries.

    Args:
        partitioned: The partitioned GraphModule from partition_by_predicate()
        full_model_inputs: Example inputs for the full model

    Returns:
        Dictionary mapping partition submodule names to their captured input tuples

    Example:
        ```python
        inputs_dict = get_example_inputs_for_partitions(partitioned, (example_tensor,))
        # {'submod_compiled_0': (example_tensor,),
        #  'submod_eager_1': (intermediate_tensor_1,),
        #  'submod_compiled_2': (intermediate_tensor_2,)}
        ```
    """
    captured_inputs = {}

    def make_capture_hook(partition_name: str):
        """Creates a hook that captures inputs for a specific partition."""

        def capture_hook(module, args):
            # Store a clone to avoid holding references to intermediate activations
            captured_inputs[partition_name] = tuple(
                arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args
            )

        return capture_hook

    handles = []
    for name, submod in partitioned.named_children():
        if name.startswith("submod_"):
            handle = submod.register_forward_pre_hook(make_capture_hook(name))
            handles.append(handle)

    with torch.no_grad():
        partitioned(*full_model_inputs)

    for handle in handles:
        handle.remove()

    return captured_inputs


class CompiledPartitionModule(torch.nn.Module):
    """
    Wrapper around TorchLikeIreeModule to make it a proper torch.nn.Module.

    This is necessary because TorchLikeIreeModule is not a torch.nn.Module,
    but we need to use it as a child module in a GraphModule.
    """

    def __init__(self, iree_module: "TorchLikeIreeModule"):
        super().__init__()
        self.iree_module = iree_module

    def forward(self, *args, **kwargs):
        """Forward pass through the IREE module."""
        result = self.iree_module.forward(*args, **kwargs)

        # TorchLikeIreeModule returns a tuple or list of tensors
        # Unwrap single-element sequences for convenience
        # TODO: DO NOT SUBMIT: This can be fixed after the modulify PR lands
        if isinstance(result, (tuple, list)) and len(result) == 1:
            return result[0]
        return result


def compile_and_replace_partitions(
    partitioned: GraphModule,
    partition_example_inputs: Dict[str, Tuple[Any, ...]],
    iree_devices: List["iree.runtime.HalDevice"],
    compile_flags: Optional[List[str]] = None,
) -> GraphModule:
    """
    Compiles 'compiled' partitions to IREE and replaces them with TorchLikeIreeModule instances.

    This function:
    1. Identifies compiled partitions (submodules named 'submod_compiled_*')
    2. Compiles each partition to IREE
    3. Loads the compiled module into IREE runtime
    4. Replaces the PyTorch submodule with a TorchLikeIreeModule

    After this function, the partitioned graph can execute with mixed eager/compiled:
    - Eager partitions run in eager PyTorch
    - Compiled partitions run in compiled IREE

    Args:
        partitioned: Partitioned GraphModule from partition_by_predicate()
        partition_example_inputs: Example inputs for each partition from
            get_example_inputs_for_partitions()
        iree_devices: List of IREE HalDevice instances for execution
        compile_flags: Optional IREE compilation flags (e.g., for HIP target)

    Returns:
        The same partitioned GraphModule, but with compiled partitions replaced by IREE modules

    Example:
        ```python
        from sharktank.utils.iree import get_iree_devices, get_iree_compiler_flags

        # Get IREE devices
        iree_devices = get_iree_devices(device='hip', device_count=1)
        compile_flags = get_iree_compiler_flags('hip', iree_hip_target='gfx942')

        # Compile and replace
        compile_and_replace_partitions(
            partitioned,
            partition_inputs,
            iree_devices,
            compile_flags
        )

        # Now partitioned module uses IREE for compiled parts
        result = partitioned(input_tensor)
        ```
    """
    from iree.turbine.aot import FxProgramsBuilder
    import iree.turbine.aot as aot
    from sharktank.utils.iree import load_iree_module, TorchLikeIreeModule

    if compile_flags is None:
        compile_flags = []

    # TODO: DO NOT SUBMIT: This can be simplified after the modulify PR lands
    for name, submodule in list(partitioned.named_children()):
        if name.startswith("submod_compiled_"):
            if name not in partition_example_inputs:
                raise ValueError(
                    f"No example inputs found for partition '{name}'. "
                    f"Available: {list(partition_example_inputs.keys())}"
                )

            example_inputs = partition_example_inputs[name]

            fxb = FxProgramsBuilder(submodule)

            @fxb.export_program(name="forward", args=example_inputs, strict=False)
            def _export(module, *args):
                return module(*args)

            export_output = aot.export(fxb)
            export_output.session.set_flags(*compile_flags)
            module_bytes = export_output.compile(save_to=None).map_memory()

            vm_module, vm_context, vm_instance = load_iree_module(
                module_buff=module_bytes, devices=iree_devices
            )

            iree_module = TorchLikeIreeModule(vm_module, vm_context, iree_devices)
            wrapped_module = CompiledPartitionModule(iree_module)

            setattr(partitioned, name, wrapped_module)

    return partitioned


def create_mixed_execution_model(
    module: torch.nn.Module,
    example_inputs: Tuple[Any, ...],
    iree_devices: List["iree.runtime.HalDevice"],
    compile_flags: Optional[List[str]] = None,
    should_run_eager_fn: Optional[Callable[[torch.nn.Module], bool]] = None,
) -> GraphModule:
    """
    Convenience function that traces, partitions, and compiles a model in one step.

    Args:
        module: The PyTorch module to convert to mixed execution
        example_inputs: Example inputs for the full model (used for tracing and compilation)
        iree_devices: List of IREE HalDevice instances for execution
        compile_flags: Optional IREE compilation flags
        should_run_eager_fn: Optional predicate function(module) -> bool.
            If None, uses default_should_run_eager which checks for @eager_mode decorator.

    Returns:
        A partitioned GraphModule with compiled partitions replaced by IREE modules

    Example:
        ```python
        from sharktank.utils.mixed_execution import create_mixed_execution_model, eager_mode
        from sharktank.utils.iree import get_iree_devices, get_iree_compiler_flags

        @eager_mode
        class EagerLayer(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.eager = EagerLayer()
                self.linear2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                x = self.linear1(x)
                x = self.eager(x)
                x = self.linear2(x)
                return x

        model = MyModel()
        example_input = torch.randn(4, 10)

        # Get IREE setup
        iree_devices = get_iree_devices(device='local-task', device_count=1)
        compile_flags = get_iree_compiler_flags('local-task')

        # Create mixed execution model
        mixed_model = create_mixed_execution_model(
            model,
            (example_input,),
            iree_devices,
            compile_flags
        )

        # Use it like a normal model
        output = mixed_model(example_input)
        ```
    """
    if should_run_eager_fn is None:
        should_run_eager_fn = default_should_run_eager

    traced = trace_module(module, should_run_eager_fn)
    partitioned = partition_by_predicate(traced, module, should_run_eager_fn)
    partition_inputs = get_example_inputs_for_partitions(partitioned, example_inputs)
    compile_and_replace_partitions(partitioned, partition_inputs, iree_devices, compile_flags)

    return partitioned


def print_partition_summary(partitioned: GraphModule) -> None:
    """
    Prints a summary of the partitioned graph structure.

    Useful for debugging and understanding the partition layout.

    Args:
        partitioned: Partitioned GraphModule from partition_by_predicate()

    Example output:
        ```
        === Partition Summary ===
        Top-level orchestrator code:
        def forward(self, x):
            submod_compiled_0 = self.submod_compiled_0(x)
            submod_eager_1 = self.submod_eager_1(submod_compiled_0)
            submod_compiled_2 = self.submod_compiled_2(submod_eager_1)
            return submod_compiled_2

        Partitions:
          - submod_compiled_0 (Compiled): 2 operations
          - submod_eager_1 (Eager): 1 operations
          - submod_compiled_2 (Compiled): 2 operations
        ```
    """
    print("\n=== Partition Summary ===")
    print("\nTop-level orchestrator code:")
    print(partitioned.code)

    print("\nPartitions:")
    for name, submod in partitioned.named_children():
        if name.startswith("submod_"):
            mode = "Compiled" if "compiled" in name else "Eager"
            if isinstance(submod, GraphModule):
                op_count = len(list(submod.graph.nodes))
                print(f"  - {name} ({mode}): {op_count} operations")
            elif isinstance(submod, CompiledPartitionModule):
                print(f"  - {name} ({mode}): IREE compiled")
            else:
                print(f"  - {name} ({mode}): {type(submod).__name__}")

    print()