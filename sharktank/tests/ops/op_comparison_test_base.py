# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Base class for op comparison testing."""

import unittest
from dataclasses import dataclass, field
import inspect
from typing import Callable, Dict, List, Any, Optional, Tuple, Union
import torch

from sharktank.ops.utils import (
    get_all_implementations,
    cast_to_type_spec,
)
from sharktank.ops._registry import _matches
from sharktank.types import (
    StaticScaledQuantizer,
    DynamicFp4BlockQuantizer,
)
from sharktank.types.layouts import (
    TensorScaledLayout,
    BlockScaledFp4Layout,
)
from sharktank.utils.testing import assert_tensor_close
from sharktank.types.tensors import unbox_tensor


@dataclass
class OpTestConfig:
    """Configuration for testing op implementations.

    Attributes:
        op: The op from sharktank.ops (e.g., ops.scaled_dot_product_attention)
        reference_impl: Direct function reference to the reference implementation
        test_impls: List of implementations to test, or "all" to auto-discover all.
        args: List of arguments to pass to the op (tensors or None for optional args)
        kwargs: Additional keyword arguments to pass to the op
        comparison_fn: Function to compare outputs (ref_output, test_output) -> None
                      Should raise AssertionError if outputs don't match
        fail_on_not_implemented: If True, fail test when implementation returns NotImplemented. If False, skip.
    """

    op: Callable
    reference_impl: Callable
    test_impls: Optional[Union[List[Callable], str]] = "all"
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    comparison_fn: Callable[[Any, Any], None] = lambda ref, test: assert_tensor_close(
        test, ref, rtol=1e-3, atol=1e-3
    )
    fail_on_not_implemented: bool = True


class OpComparisonTestBase(unittest.TestCase):
    """Base class for comparing op implementations."""

    def _get_override_type_spec(self, op, override_func):
        """Get the type spec for an override function."""
        for override in op._overrides:
            if override.target == override_func:
                return override.type_spec
        raise ValueError(f"Could not find type spec for {override_func.__name__}")

    LAYOUT_TO_QUANTIZER = {
        TensorScaledLayout: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=dtype
        ),
        BlockScaledFp4Layout: lambda dtype=None: DynamicFp4BlockQuantizer(
            block_size=32,
        ),
        # TODO: Still need suitable default quantizers for:
        # BlockScaledLayout, BlockScaledI4Layout, SuperBlockOffsetScaled_4_6_Layout
    }

    def cast_inputs_for_override(
        self, op: Callable, override_func: Callable, args: List[Any]
    ) -> List[Any]:
        """Cast inputs to match override signature types.

        Args:
            override_func: The override function
            args: List of input values
            config: Test configuration

        Returns:
            List of inputs cast to appropriate types
        """
        type_spec = self._get_override_type_spec(op, override_func)

        # Extract layout types if the function uses @quantized_tensor_layout_of_type
        layout_types = None
        if hasattr(override_func, "_layout_types"):
            layout_types = self._extract_layout_types_from_decorator(
                override_func, args
            )

        return cast_to_type_spec(
            args, type_spec, self.LAYOUT_TO_QUANTIZER, layout_types
        )

    def _extract_layout_types_from_decorator(
        self, func: Callable, args: List[Any]
    ) -> Optional[Tuple[type, ...]]:
        """Extract layout types from @quantized_tensor_layout_of_type decorator.

        Returns a tuple of layout types corresponding to the function parameters.
        """

        layout_dict = func._layout_types
        if layout_dict:
            # Get parameter names from the original function
            original_func = func.__wrapped__ if hasattr(func, "__wrapped__") else func
            sig = inspect.signature(original_func)
            param_names = list(sig.parameters.keys())
            # Return layout types in parameter order
            return tuple(layout_dict.get(name) for name in param_names[: len(args)])

        return None

    def compare_outputs(
        self,
        reference_output: Any,
        test_output: Any,
        config: OpTestConfig,
        impl_name: str,
    ):
        """Compare two outputs using the configured comparison function.

        Args:
            reference_output: Reference output
            test_output: Test output
            config: Test configuration
            impl_name: Name of the implementation being tested
        """

        reference_output = unbox_tensor(reference_output)
        test_output = unbox_tensor(test_output)

        try:
            config.comparison_fn(reference_output, test_output)
        except AssertionError as e:
            ref_name = config.reference_impl.__name__
            raise AssertionError(
                f"Implementation '{impl_name}' failed comparison against reference '{ref_name}': {e}"
            )

    def compare_implementations(self, config: OpTestConfig):
        """Main comparison method that tests all implementations.

        Args:
            config: Test configuration
        """
        all_impls = get_all_implementations(config.op)

        if not config.reference_impl:
            self.fail("No reference implementation specified")

        ref_name = config.reference_impl.__name__

        ref_args = self.cast_inputs_for_override(
            config.op, config.reference_impl, config.args
        )
        ref_output = config.reference_impl(*ref_args, **config.kwargs)

        if ref_output is NotImplemented:
            self.fail(f"Reference implementation '{ref_name}' returned NotImplemented")

        if config.test_impls != "all":
            test_impls = {func.__name__: func for func in config.test_impls}
        else:
            # Test all discovered implementations except the reference
            # TODO: Add support for testing sharded implementations by creating
            # appropriate sharded tensor inputs with distribution context
            test_impls = {}
            for name, func in all_impls.items():
                if name == ref_name:
                    continue
                # Skip sharded implementations for now
                type_spec = self._get_override_type_spec(config.op, func)
                from sharktank.types import (
                    SplitPrimitiveTensor,
                    ReplicatedTensor,
                    UnreducedTensor,
                )

                has_sharded = any(
                    _matches(t, SplitPrimitiveTensor)
                    or _matches(t, ReplicatedTensor)
                    or _matches(t, UnreducedTensor)
                    for t in type_spec
                    if t is not None
                )
                if has_sharded:
                    continue
                test_impls[name] = func

        for impl_name in sorted(test_impls.keys()):
            impl_func = test_impls[impl_name]

            with self.subTest(implementation=impl_name):
                impl_args = self.cast_inputs_for_override(
                    config.op, impl_func, config.args
                )
                impl_output = impl_func(*impl_args, **config.kwargs)

                if impl_output is NotImplemented:
                    if config.fail_on_not_implemented:
                        self.fail(
                            f"Implementation '{impl_name}' returned NotImplemented"
                        )
                    else:
                        continue

                self.compare_outputs(ref_output, impl_output, config, impl_name)
