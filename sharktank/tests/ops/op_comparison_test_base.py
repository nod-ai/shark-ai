# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Base class for op comparison testing."""

import unittest
from typing import Callable, Dict, List, Any, Optional
import torch
import numpy as np

from sharktank.ops.testing_utils import (
    get_all_implementations,
    get_override_type_spec,
    cast_to_type_spec,
)
from sharktank.ops._registry import _matches
from sharktank.types import (
    DynamicScaledQuantizer,
    StaticScaledQuantizer,
    DynamicFp4BlockQuantizer,
)
from sharktank.types.layouts import (
    TensorScaledLayout,
    BlockScaledLayout,
    BlockScaledFp4Layout,
    BlockScaledI4Layout,
    SuperBlockOffsetScaled_4_6_Layout,
)
from sharktank.utils.testing import assert_tensor_close
from sharktank.utils.math import cosine_similarity
from .op_test_configs import OpTestConfig


class OpComparisonTestBase(unittest.TestCase):
    """Base class for comparing op implementations."""

    # Layout to quantizer mapping for automatic quantization
    # Maps layout types to quantizer functions that can create quantized tensors
    LAYOUT_TO_QUANTIZER = {
        # TensorScaledLayout: Use scale=1 to avoid extreme values in testing
        TensorScaledLayout: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=dtype
        ),
        # BlockScaledLayout: Use scale=1 for testing
        BlockScaledLayout: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=dtype
        ),
        # BlockScaledFp4Layout: FP4 block quantization
        BlockScaledFp4Layout: lambda dtype=None: DynamicFp4BlockQuantizer(
            block_size=32,
        ),
        # BlockScaledI4Layout: INT4 block quantization
        BlockScaledI4Layout: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=torch.int8
        ),
        # SuperBlockOffsetScaled_4_6_Layout: Use scale=1 for testing
        SuperBlockOffsetScaled_4_6_Layout: lambda dtype: StaticScaledQuantizer(
            scale=torch.tensor(1.0), dtype=dtype
        ),
    }

    def discover_implementations(self, op) -> Dict[str, Callable]:
        """Automatically discover all registered implementations for an op.

        Args:
            op: The op to discover implementations for

        Returns:
            Dictionary mapping implementation names to functions
        """
        return get_all_implementations(op)

    def cast_inputs_for_override(
        self, op, override_func: Callable, args: List[Any], config: OpTestConfig
    ) -> List[Any]:
        """Cast inputs to match override signature types.

        Args:
            op: The op dispatcher
            override_func: The override function
            args: List of input values
            config: Test configuration

        Returns:
            List of inputs cast to appropriate types
        """
        type_spec = get_override_type_spec(op, override_func)
        if type_spec is None:
            # If we can't find the type spec, just return args as-is
            return args

        return cast_to_type_spec(args, type_spec, self.LAYOUT_TO_QUANTIZER)

    def compare_outputs(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        config: OpTestConfig,
        impl_name: str,
    ):
        """Compare two outputs using the specified method.

        Args:
            output1: Reference output
            output2: Test output
            config: Test configuration
            impl_name: Name of the implementation being tested
        """
        # Ensure both outputs are tensors
        from sharktank.types.tensors import unbox_tensor

        output1 = unbox_tensor(output1)
        output2 = unbox_tensor(output2)

        if config.comparison_method == "assert_close":
            try:
                assert_tensor_close(
                    output2, output1, rtol=config.rtol, atol=config.atol
                )
            except AssertionError as e:
                raise AssertionError(
                    f"Implementation '{impl_name}' failed assert_close: {e}"
                )

        elif config.comparison_method == "cosine_similarity":
            similarity = cosine_similarity(output1.flatten(), output2.flatten())
            if similarity < config.cosine_similarity_threshold:
                raise AssertionError(
                    f"Implementation '{impl_name}' failed cosine similarity: "
                    f"{similarity:.6f} < {config.cosine_similarity_threshold}"
                )

        elif config.comparison_method == "both":
            # Try assert_close first
            try:
                assert_tensor_close(
                    output2, output1, rtol=config.rtol, atol=config.atol
                )
            except AssertionError:
                # Fall back to cosine similarity
                similarity = cosine_similarity(output1.flatten(), output2.flatten())
                if similarity < config.cosine_similarity_threshold:
                    raise AssertionError(
                        f"Implementation '{impl_name}' failed both comparisons. "
                        f"Cosine similarity: {similarity:.6f} < {config.cosine_similarity_threshold}"
                    )
        else:
            raise ValueError(f"Unknown comparison method: {config.comparison_method}")

    def compare_implementations(self, config: OpTestConfig):
        """Main comparison method that tests all implementations.

        Args:
            config: Test configuration
        """
        # Discover all implementations for this op
        all_impls = self.discover_implementations(config.op)

        # Determine reference implementation
        if not config.reference_impl:
            self.skipTest("No reference implementation specified")

        ref_name = config.reference_impl.__name__

        # First, run the reference implementation
        ref_args = self.cast_inputs_for_override(
            config.op, config.reference_impl, config.args, config
        )
        ref_output = config.reference_impl(*ref_args, **config.kwargs)

        if ref_output is NotImplemented:
            self.skipTest(
                f"Reference implementation '{ref_name}' returned NotImplemented"
            )

        # Determine which implementations to test
        if config.test_impls is not None:
            # Specific implementations requested
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
                type_spec = get_override_type_spec(config.op, func)
                if type_spec:
                    # Check if any type in the spec is a sharded type
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

        # Generate a subtest for each implementation
        for impl_name in sorted(test_impls.keys()):
            impl_func = test_impls[impl_name]

            with self.subTest(implementation=impl_name):
                # Cast inputs for this implementation
                impl_args = self.cast_inputs_for_override(
                    config.op, impl_func, config.args, config
                )

                # Run implementation
                try:
                    impl_output = impl_func(*impl_args, **config.kwargs)

                    # Check if implementation returned NotImplemented
                    if impl_output is NotImplemented:
                        if config.fail_on_not_implemented:
                            self.fail(
                                f"Implementation '{impl_name}' returned NotImplemented"
                            )
                        else:
                            # Continue to next implementation without failing
                            continue

                    # Compare against reference
                    self.compare_outputs(ref_output, impl_output, config, impl_name)

                except unittest.SkipTest:
                    raise
                except Exception as e:
                    raise AssertionError(
                        f"Implementation '{impl_name}' raised exception: {e}"
                    )
