# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from unittest import TestCase

from sharktank.utils.mixed_execution import (
    eager_mode,
    trace_module,
    partition_with_transitions,
    get_example_inputs_for_partitions,
    compile_and_replace_partitions,
    create_mixed_execution_model,
    print_partition_summary,
    default_should_run_eager,
)
from sharktank.utils.iree import get_iree_devices, get_iree_compiler_flags


@eager_mode
class EagerOnlyLayer(torch.nn.Module):
    """A leaf module that should run in eager mode."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple operation that we want to keep in eager mode
        return x.sign() * x.abs().clamp(min=0, max=1)


class SimpleModel(torch.nn.Module):
    """
    Simple test model with mixed eager/compiled operations.

    Structure:
        linear1 (compiled) → eager_layer (eager) → linear2 (compiled)
    """

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.eager_layer = EagerOnlyLayer()
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize with small weights for numerical stability
        torch.nn.init.normal_(self.linear1.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.eager_layer(x)
        x = self.linear2(x)
        return x


class TestPartitioning(TestCase):
    """Tests for graph partitioning without IREE compilation."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 16
        self.batch_size = 4
        self.model = SimpleModel(self.hidden_size)
        self.example_input = torch.randn(self.batch_size, self.hidden_size)

    def test_partition_structure(self):
        """Test that partitioning creates the expected submodule structure."""
        traced = trace_module(self.model)
        partitioned = partition_with_transitions(traced, self.model, default_should_run_eager)

        # Check that we have the expected partitions
        partition_names = [
            name for name in dict(partitioned.named_children()).keys()
            if name.startswith("submod_")
        ]

        # Should have 3 partitions: compiled_0, eager_1, compiled_2
        self.assertEqual(len(partition_names), 3)
        self.assertIn("submod_compiled_0", partition_names)
        self.assertIn("submod_eager_1", partition_names)
        self.assertIn("submod_compiled_2", partition_names)

    def test_partition_execution_eager_only(self):
        """Test that partitioned model produces same results as original (eager mode)."""
        traced = trace_module(self.model)
        partitioned = partition_with_transitions(traced, self.model, default_should_run_eager)

        # Run both models
        with torch.no_grad():
            expected_output = self.model(self.example_input)
            actual_output = partitioned(self.example_input)

        # Results should match
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)

    def test_example_input_capture(self):
        """Test that example input capture works correctly."""
        # Trace and partition using simplified API
        traced = trace_module(self.model)
        partitioned = partition_with_transitions(traced, self.model, default_should_run_eager)

        # Capture example inputs
        captured = get_example_inputs_for_partitions(
            partitioned, (self.example_input,)
        )

        # Should have captured inputs for all 3 partitions
        self.assertEqual(len(captured), 3)
        self.assertIn("submod_compiled_0", captured)
        self.assertIn("submod_eager_1", captured)
        self.assertIn("submod_compiled_2", captured)

        # First partition should get the original input
        self.assertEqual(len(captured["submod_compiled_0"]), 1)
        torch.testing.assert_close(
            captured["submod_compiled_0"][0], self.example_input
        )

        # Other partitions should have intermediate tensors
        for name in ["submod_eager_1", "submod_compiled_2"]:
            self.assertEqual(len(captured[name]), 1)
            self.assertEqual(
                captured[name][0].shape, (self.batch_size, self.hidden_size)
            )


@pytest.mark.usefixtures("iree_flags")
class TestMixedExecution(TestCase):
    """Tests for mixed eager/compiled execution with IREE."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_size = 16
        self.batch_size = 4
        self.model = SimpleModel(self.hidden_size)
        self.example_input = torch.randn(self.batch_size, self.hidden_size)

    def test_mixed_execution_with_iree(self):
        """Test that mixed execution with IREE compilation produces correct results."""
        # Get original output for comparison
        with torch.no_grad():
            expected_output = self.model(self.example_input)

        # Get IREE devices and compiler flags
        iree_devices = get_iree_devices(device=self.iree_device, device_count=1)
        compile_flags = get_iree_compiler_flags(
            self.iree_hal_target_device,
            self.iree_hal_local_target_device_backends
            if hasattr(self, "iree_hal_local_target_device_backends")
            else None,
        )

        mixed_model = create_mixed_execution_model(
            self.model,
            (self.example_input,),
            iree_devices,
            compile_flags
        )

        # Print summary for debugging
        print_partition_summary(mixed_model)

        # Run with mixed execution
        with torch.no_grad():
            actual_output = mixed_model(self.example_input)

        # Results should match original model
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-4, atol=1e-4)


class TestEdgeCases(TestCase):
    """Tests for edge cases in partitioning."""

    def test_all_eager_partitioning(self):
        """Test partitioning when everything should run eager."""
        model = SimpleModel(hidden_size=8)
        traced = trace_module(model)

        # Everything runs eager
        def should_run_eager(module):
            return True

        partitioned = partition_with_transitions(traced, model, should_run_eager)

        # Should have a single eager partition
        partition_names = [
            name for name in dict(partitioned.named_children()).keys()
            if name.startswith("submod_")
        ]
        self.assertEqual(len(partition_names), 1)
        self.assertEqual(partition_names[0], "submod_eager_0")

    def test_all_compiled_partitioning(self):
        """Test partitioning when everything should be compiled."""
        model = SimpleModel(hidden_size=8)
        traced = trace_module(model)

        # Everything runs compiled
        def should_run_eager(module):
            return False

        partitioned = partition_with_transitions(traced, model, should_run_eager)

        # Should have a single compiled partition
        partition_names = [
            name for name in dict(partitioned.named_children()).keys()
            if name.startswith("submod_")
        ]
        self.assertEqual(len(partition_names), 1)
        self.assertEqual(partition_names[0], "submod_compiled_0")

    def test_print_partition_summary(self):
        """Test that print_partition_summary doesn't crash."""
        model = SimpleModel(hidden_size=8)
        traced = trace_module(model)
        partitioned = partition_with_transitions(traced, model, default_should_run_eager)

        # Should not crash
        print_partition_summary(partitioned)