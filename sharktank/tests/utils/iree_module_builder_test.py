# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for high-level IREE module builder utilities."""

import pytest
import torch
import torch.nn as nn

from sharktank.utils.iree_module_builder import (
    compile_torch_module_to_iree,
    load_torch_module_as_iree,
    oneshot_compile_and_run,
)
from sharktank.utils.iree import TypePreservingIreeModule, TorchLikeIreeModule
from sharktank.types import SplitPrimitiveTensor


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MultiOutputModel(nn.Module):
    """Model that returns multiple outputs."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)

    def forward(self, x):
        h = self.fc(x)
        return torch.relu(h), torch.tanh(h)


class TestCompileTorchModule:
    """Tests for compile_torch_module_to_iree."""

    def test_basic_compilation(self):
        """Test basic compilation without saving artifacts."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)

        vmfb_bytes = compile_torch_module_to_iree(
            model,
            example_args=(example_input,),
            compile_args=["--iree-hal-target-device=local-task"],
        )

        assert isinstance(vmfb_bytes, memoryview)
        assert len(vmfb_bytes) > 0

    def test_compilation_with_save_mlir(self, tmp_path):
        """Test compilation with MLIR saving."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)
        mlir_path = tmp_path / "model.mlir"

        vmfb_bytes = compile_torch_module_to_iree(
            model,
            example_args=(example_input,),
            compile_args=["--iree-hal-target-device=local-task"],
            save_mlir_to=mlir_path,
        )

        assert mlir_path.exists()
        assert mlir_path.stat().st_size > 0
        assert isinstance(vmfb_bytes, memoryview)

    def test_compilation_with_save_vmfb(self, tmp_path):
        """Test compilation with VMFB saving."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)
        vmfb_path = tmp_path / "model.vmfb"

        vmfb_bytes = compile_torch_module_to_iree(
            model,
            example_args=(example_input,),
            compile_args=["--iree-hal-target-device=local-task"],
            save_vmfb_to=vmfb_path,
        )

        assert vmfb_path.exists()
        assert vmfb_path.stat().st_size > 0
        assert isinstance(vmfb_bytes, memoryview)

    def test_compilation_with_kwargs(self):
        """Test compilation with keyword arguments."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)

        vmfb_bytes = compile_torch_module_to_iree(
            model,
            example_args=tuple(),
            example_kwargs={"x": example_input},
            compile_args=["--iree-hal-target-device=local-task"],
        )

        assert isinstance(vmfb_bytes, memoryview)
        assert len(vmfb_bytes) > 0


class TestLoadTorchModuleAsIree:
    """Tests for load_torch_module_as_iree."""

    def test_basic_loading_and_execution(self):
        """Test that loaded module executes and produces correct output shape."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)

        iree_module = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
        )

        result = iree_module.forward(example_input)
        # Verify output structure
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (2, 10)
        # Verify it's actually a tensor with values
        assert not torch.isnan(result[0]).any()

    def test_output_matches_torch(self):
        """Test that IREE output matches torch output."""
        torch.manual_seed(42)
        model = SimpleModel()
        model.eval()
        example_input = torch.randn(2, 32)

        # Get torch output
        with torch.no_grad():
            torch_output = model(example_input)

        # Get IREE output
        iree_module = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
        )
        iree_output = iree_module.forward(example_input)

        # Compare
        torch.testing.assert_close(iree_output[0], torch_output, rtol=1e-4, atol=1e-4)

    def test_multi_output_model(self):
        """Test model with multiple outputs."""
        model = MultiOutputModel()
        example_input = torch.randn(2, 32)

        iree_module = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
        )

        result = iree_module.forward(example_input)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (2, 64)
        assert result[1].shape == (2, 64)


class TestTypePreservingIreeModule:
    """Tests for TypePreservingIreeModule and output_type_mapper."""

    def test_output_type_mapper_changes_return_structure(self):
        """Test that output_type_mapper successfully transforms return type."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)

        # Without type mapper - returns tuple
        iree_module = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
        )
        result_tuple = iree_module.forward(example_input)

        # With type mapper - returns unwrapped single tensor
        def unwrap(outputs):
            return outputs[0]

        iree_module_unwrapped = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
            output_type_mapper=unwrap,
        )
        result_single = iree_module_unwrapped.forward(example_input)

        # Verify the transformation worked
        assert isinstance(result_tuple, tuple)
        assert isinstance(result_single, torch.Tensor)
        assert not isinstance(result_single, tuple)
        torch.testing.assert_close(result_single, result_tuple[0])

    def test_reconstruct_sharded_tensor(self):
        """Test reconstructing a ShardedTensor-like output."""
        model = MultiOutputModel()
        example_input = torch.randn(2, 32)

        # Simulate sharded output reconstruction
        def reconstruct_sharded(outputs):
            # Treat the two outputs as shards
            return SplitPrimitiveTensor(ts=outputs, shard_dim=1)

        iree_module = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
            output_type_mapper=reconstruct_sharded,
        )

        result = iree_module.forward(example_input)
        assert isinstance(result, SplitPrimitiveTensor)
        assert result.shard_dim == 1
        assert len(result.shards) == 2

    def test_custom_transformation(self):
        """Test custom output transformation."""
        model = MultiOutputModel()
        example_input = torch.randn(2, 32)

        # Custom transformer: return dict
        def to_dict(outputs):
            return {"relu": outputs[0], "tanh": outputs[1]}

        iree_module = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
            output_type_mapper=to_dict,
        )

        result = iree_module.forward(example_input)
        assert isinstance(result, dict)
        assert "relu" in result
        assert "tanh" in result
        assert result["relu"].shape == (2, 64)
        assert result["tanh"].shape == (2, 64)


class TestOneshotCompileAndRun:
    """Tests for oneshot_compile_and_run."""

    def test_basic_oneshot(self):
        """Test basic one-shot execution."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)

        result = oneshot_compile_and_run(
            model,
            args=(example_input,),
            device="local-task",
            compile_args=("--iree-hal-target-device=local-task",),
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (2, 10)

    def test_oneshot_matches_torch(self):
        """Test that one-shot execution matches torch."""
        torch.manual_seed(42)
        model = SimpleModel()
        model.eval()
        example_input = torch.randn(2, 32)

        # Torch output
        with torch.no_grad():
            torch_output = model(example_input)

        # IREE output
        iree_output = oneshot_compile_and_run(
            model,
            args=(example_input,),
            device="local-task",
            compile_args=("--iree-hal-target-device=local-task",),
        )

        torch.testing.assert_close(iree_output[0], torch_output, rtol=1e-4, atol=1e-4)

    def test_oneshot_with_kwargs(self):
        """Test one-shot with keyword arguments."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)

        result = oneshot_compile_and_run(
            model,
            args=tuple(),
            kwargs={"x": example_input},
            device="local-task",
            compile_args=("--iree-hal-target-device=local-task",),
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (2, 10)


class TestInferenceModuleProtocol:
    """Tests for Protocol-based usage."""

    def test_protocol_compatibility(self):
        """Test that both torch and IREE modules work with generic inference code."""
        from sharktank.utils.inference_module import InferenceModule

        def run_model(model: InferenceModule, input_data):
            """Generic function that works with any InferenceModule."""
            return model.forward(input_data)

        torch_model = SimpleModel()
        example_input = torch.randn(2, 32)

        # Works with torch module
        torch_model.eval()
        with torch.no_grad():
            torch_result = run_model(torch_model, example_input)
        assert torch_result.shape == (2, 10)

        # Works with IREE module
        iree_model = load_torch_module_as_iree(
            torch_model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
            output_type_mapper=lambda x: x[0],  # Unwrap for compatibility
        )
        iree_result = run_model(iree_model, example_input)
        assert iree_result.shape == (2, 10)

        # Results should be close
        torch.testing.assert_close(iree_result, torch_result, rtol=1e-4, atol=1e-4)

    def test_call_vs_forward_equivalence(self):
        """Test that calling via __call__ produces same result as forward()."""
        model = SimpleModel()
        example_input = torch.randn(2, 32)

        iree_model = load_torch_module_as_iree(
            model,
            example_args=(example_input,),
            device="local-task",
            compile_args=["--iree-hal-target-device=local-task"],
            output_type_mapper=lambda x: x[0],
        )

        result1 = iree_model(example_input)
        result2 = iree_model.forward(example_input)

        torch.testing.assert_close(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
