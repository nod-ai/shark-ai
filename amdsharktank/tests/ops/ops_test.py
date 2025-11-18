# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from itertools import product
from numbers import Number
from pathlib import Path
from typing import Callable
import unittest

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import iree.turbine.aot as aot
from iree.turbine.aot import FxProgramsBuilder
import iree.runtime
import iree.compiler
from parameterized import parameterized, parameterized_class
import safetensors
from amdsharktank import ops
from amdsharktank.types import *
from amdsharktank.layers import BaseLayer
from amdsharktank.utils import debugging
from amdsharktank.utils.testing import (
    TempDirTestBase,
    assert_tensor_close,
    create_sample_tensor_from_class,
)
from amdsharktank.utils.iree import (
    with_iree_device_context,
    get_iree_compiler_flags_from_object,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    make_hal_buffer_view_trace_default_callback,
    oneshot_iree_run,
)


class AbsTest(unittest.TestCase):
    def testAbsPrimitiveTensor(self):
        tensor = torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float32)
        expected_result = torch.abs(tensor)
        actual_result = DefaultPrimitiveTensor(data=tensor).abs()
        assert_tensor_close(actual_result, expected_result)


class ArangeTest(unittest.TestCase):
    def testEndOnly(self):
        expected = torch.arange(5)
        actual = ops.arange(5)
        assert ops.equal(expected, actual)

    def testStartEnd(self):
        expected = torch.arange(2, 7)
        actual = ops.arange(2, 7)
        assert ops.equal(expected, actual)

    def testStartEndStep(self):
        expected = torch.arange(1, 10, 2)
        actual = ops.arange(1, 10, 2)
        assert ops.equal(expected, actual)

    @parameterized.expand([torch.float32, torch.int64])
    def testDtype(self, dtype):
        expected = torch.arange(0, 6, dtype=dtype)
        actual = ops.arange(0, 6, dtype=dtype)
        assert expected.dtype == actual.dtype
        assert ops.equal(expected, actual)

    def testFloatStep(self):
        expected = torch.arange(0.0, 1.0, 0.1, dtype=torch.float32)
        actual = ops.arange(0.0, 1.0, 0.1, dtype=torch.float32)
        assert ops.equal(expected, actual)

    def testEmptyRange(self):
        expected = torch.arange(5, 5)
        actual = ops.arange(5, 5)
        assert ops.equal(expected, actual)

    @parameterized.expand(
        [
            ((3,),),
            ((3, 2),),
            ((3, 2, 1),),
        ]
    )
    def testDevicesReplicated(self, devices: tuple[int, ...]):
        expected = torch.arange(0, 6, dtype=torch.float32)
        actual = ops.arange(0, 6, devices=devices, dtype=torch.float32)
        assert isinstance(actual, ReplicatedTensor)
        assert tuple(devices) == actual.devices
        assert actual.shard_count == len(devices)
        assert all(ops.equal(shard, expected) for shard in actual.shards)


class ArgmaxTest(unittest.TestCase):
    @parameterized.expand([torch.float16, torch.float32])
    def testArgmax(self, dtype):
        a = torch.zeros(1, 1, 256, dtype=dtype)
        a[0][0][42] = 42
        assert ops.argmax(a, -1) == 42

    @parameterized.expand([torch.float16, torch.float32])
    def testArgmaxDim0(self, dtype):
        a = torch.zeros(3, 1, 256, dtype=dtype)
        a[1][0][42] = 42
        result = ops.argmax(a, 0)
        assert result[0][42] == 1

    @parameterized.expand([torch.float16, torch.float32])
    def testArgmaxKeepdim(self, dtype):
        a = torch.zeros(2, 4, dtype=dtype)
        a[1][0] = 42
        a[1][2] = 99
        a[0][1] = 1
        a[0][3] = 1
        result = ops.argmax(a, 0, True)
        expected = torch.tensor([[1, 0, 1, 0]], dtype=torch.int64)
        assert result.shape == (1, 4)
        assert ops.equal(result, expected)

    @parameterized.expand(
        [
            torch.float16,
            torch.float32,
        ]
    )
    def testSplitArgmax(self, dtype):
        a = torch.zeros(1, 1, 256, dtype=dtype)
        a[0][0][42] = 42
        assert ops.argmax(a, -1, chunk_size=16) == 42

    def testSplitArgmaxDim0(self):
        for dtype in [torch.float16, torch.float32]:
            a = torch.zeros(3, 1, 256, dtype=dtype)
            a[1][0][42] = 42
            result = ops.argmax(a, 0, chunk_size=1)
            assert result[0][42] == 1

    def testSplitArgmaxKeepdim(self):
        for dtype in [torch.float16, torch.float32]:
            a = torch.zeros(2, 4, dtype=dtype)
            a[1][0] = 42
            a[1][2] = 99
            a[0][1] = 1
            a[0][3] = 1
            result = ops.argmax(a, 0, True, 1)
            expected = torch.tensor([[1, 0, 1, 0]], dtype=torch.int64)
            assert result.shape == (1, 4)
            assert ops.equal(result, expected)

    @parameterized.expand(
        [
            ([4, 32, 131072], torch.float16),
            ([4, 32, 131072], torch.float32),
            ([32, 1, 131072], torch.float16),
            ([32, 1, 131072], torch.float32),
        ]
    )
    def testSplitArgmaxRandom(self, shape, dtype):
        a = torch.rand(*shape, dtype=dtype)
        expected = torch.argmax(a, -1)
        result = ops.argmax(a, -1, chunk_size=128)
        assert ops.equal(expected, result)

    def testSplitArgmaxRandomDim0(self):
        a = torch.rand(4, 32, 131072, dtype=torch.float16)
        expected = torch.argmax(a, 0)
        result = ops.argmax(a, 0, chunk_size=2)
        assert ops.equal(expected, result)

    def testSplitArgmaxInvalidChunkSize(self):
        a = torch.rand(4, 32, 100, dtype=torch.float32)

        with pytest.raises(ValueError):
            ops.argmax(a, 0, chunk_size=42)


class BroadcastDimsTest(unittest.TestCase):
    def testBroadcastDimForSmallerRankTensor(self):
        a = torch.empty(2, 5, 1)
        b = torch.empty(4, 2, 5, 1)
        assert ops.broadcast_dim(2, [a, b]) == 3

    def testBroadcastDimForLargestRankTensor(self):
        a = torch.empty(4, 2, 5, 1)
        b = torch.empty(2, 5, 1)
        assert ops.broadcast_dim(2, [a, b]) == 2

    def testBroadcastDims(self):
        a = torch.empty(4, 2, 1, 2)
        b = torch.empty(2, 3, 2)
        tensors = [a, b]
        dims = [0, 1]
        res = ops.broadcast_dims(dims, tensors)
        assert res[0] == 0
        assert res[1] == 2


class TestCat:
    @pytest.mark.parametrize(
        "dtype, dim", [(torch.float8_e4m3fnuz, 0), (torch.float8_e4m3fn, 1)]
    )
    def testCatEagerF8(self, deterministic_random_seed, dtype: torch.dtype, dim: int):
        tensors = [torch.rand([2, 3, 4], dtype=torch.float32) for _ in range(2)]
        tensors = [t.to(dtype=dtype) for t in tensors]
        actual = ops.cat(tensors, dim=dim)

        tensors_as_int8 = [t.view(dtype=torch.int8) for t in tensors]
        expected = torch.cat(tensors_as_int8, dim=dim).view(dtype=dtype)
        assert_tensor_close(actual, expected, rtol=0, atol=0)


class ChunkTest(unittest.TestCase):
    def testChunkBasic(self):
        """Test basic chunk operation with 2 chunks."""
        a = torch.arange(10, dtype=torch.float32)
        result_chunks = ops.chunk(a, 2)
        expected_chunks = a.chunk(2)
        assert all(
            ops.equal(actual, expected_chunk)
            for actual, expected_chunk in zip(result_chunks, expected_chunks)
        )

    def testChunkUneven(self):
        """Test chunk operation with uneven division."""
        a = torch.arange(10, dtype=torch.float32)
        expected_chunks = a.chunk(3)
        result_chunks = ops.chunk(a, 3)
        assert all(
            ops.equal(actual, expected_chunk)
            for actual, expected_chunk in zip(result_chunks, expected_chunks)
        )

    def testChunkReplicated(self):
        """Test chunk operation with replicated tensor."""
        a = torch.arange(10, dtype=torch.float32)
        replicated = ops.replicate(a, 2)
        expected_chunks = a.chunk(2)
        result_chunks = ops.chunk(replicated, 2)
        assert all(
            ops.equal(actual, expected_chunk)
            for actual, expected_chunk in zip(result_chunks, expected_chunks)
        )


class EqualTest(unittest.TestCase):
    def testEqualTorchTensors(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = torch.clone(a)
        assert ops.equal(a, b)
        assert ops.equal(b, a)

    def testNotEqualTorchTensors(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = torch.clone(a)
        b[0, 0] += 1
        assert not ops.equal(a, b)
        assert not ops.equal(b, a)

    def testEqualTorchTensorAndPrimitiveTensor(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = DefaultPrimitiveTensor(data=torch.clone(a))
        assert ops.equal(a, b)
        assert ops.equal(b, a)

    def testEqualTorchTensorAndPrimitiveTensor(self):
        a = torch.rand(2, 3, dtype=torch.float32)
        b = DefaultPrimitiveTensor(data=torch.clone(a))
        b.as_torch()[0, 0] += 1
        assert not ops.equal(a, b)
        assert not ops.equal(b, a)


class EmbeddingLookupTest(unittest.TestCase):
    def testTorchImplNoCast(self):
        t1 = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        t2 = torch.rand(10, 3, dtype=torch.float32)
        result = ops.embedding_lookup(t1, t2, torch.float32)
        expected = F.embedding(t1, t2)
        assert_tensor_close(result, expected)

    def testTorchImplCast(self):
        t1 = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        t2 = torch.rand(10, 3, dtype=torch.float16)
        result = ops.embedding_lookup(t1, t2, torch.float32)
        expected = F.embedding(t1, t2.to(torch.float32))
        assert_tensor_close(result, expected)

    def testPrimitiveTensorRhs(self):
        t1 = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        t2 = torch.rand(10, 3, dtype=torch.float16)
        t2_pt = DefaultPrimitiveTensor(data=t2)
        result = ops.embedding_lookup(t1, t2_pt, torch.float32)
        expected = F.embedding(t1, t2.to(torch.float32))
        assert_tensor_close(result, expected)

    def testQuantizedTensorRhs(self):
        # TODO: Implement me. Quantized embedding lookup NYI completely.
        ...


class FullTest(unittest.TestCase):
    def testFullBasic(self):
        size = (3, 2)
        value = 7.5
        expected = torch.full(size, value)
        actual = ops.full(size, value)
        assert expected.dtype == actual.dtype
        assert ops.equal(expected, actual)

    @parameterized.expand([torch.float32, torch.int64, torch.complex128])
    def testDtype(self, dtype):
        size = (5,)
        value = 3
        expected = torch.full(size, value, dtype=dtype)
        actual = ops.full(size, value, dtype=dtype)
        assert expected.dtype == actual.dtype
        assert ops.equal(expected, actual)


class GemmTest(unittest.TestCase):
    def testGemm(self):
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])
        c = torch.tensor([[9, 10], [11, 12]])
        alpha = 2
        beta = 3
        expected = alpha * a @ b.T + beta * c
        result = ops.gemm(a, b, c, alpha, beta, False, True)
        assert_tensor_close(result, expected)


class LogTest(unittest.TestCase):
    def testLogPrimitiveTensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        expected_result = torch.log(tensor)
        actual_result = DefaultPrimitiveTensor(data=tensor).log()
        assert_tensor_close(actual_result, expected_result)


class LogicalOrTest(unittest.TestCase):
    def testLogicalOrTorchTensors(self):
        a = torch.tensor([True, False, True, False])
        b = torch.tensor([True, True, False, False])
        expected_result = torch.logical_or(a, b)
        actual_result = ops.logical_or(a, b)
        assert_tensor_close(actual_result, expected_result)

    def testLogicalOrPrimitiveTensors(self):
        a = DefaultPrimitiveTensor(data=torch.tensor([True, False, True, False]))
        b = DefaultPrimitiveTensor(data=torch.tensor([True, True, False, False]))
        expected_result = torch.logical_or(a.as_torch(), b.as_torch())
        actual_result = ops.logical_or(a, b)
        assert_tensor_close(actual_result, expected_result)

    def testLogicalOrMixedTypes(self):
        a = torch.tensor([0, 1, 0, 1])
        b = torch.tensor([0, 0, 1, 1])
        expected_result = torch.logical_or(a, b)
        actual_result = ops.logical_or(a, b)
        assert_tensor_close(actual_result, expected_result)


class MatmulTest(unittest.TestCase):
    def tearDown(self):
        ops._registry._test_enable_last_op_dispatch(False)

    def testMatchFail(self):
        # This is just using matmul as a victim to test that failure/exceptions
        # are properly raised when no override is found.
        with self.assertRaisesRegex(
            NotImplementedError,
            r"Overridable operator.+does not have an implementation for argument types:.+int.+int",
        ):
            ops.matmul(1, 2)

    @unittest.skip("https://github.com/nod-ai/amdshark-ai/issues/44")
    def testTorchImplTransposedRHS(self):
        ops._registry._test_enable_last_op_dispatch(True)
        t1 = torch.rand(32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        result = ops.matmul(t1, t2.T)
        expected = torch.matmul(t1, t2.T.to(torch.float32))
        assert_tensor_close(result, expected)
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.custom_impls.matmul_mmtfp_tensor_tensor,
        )

    @unittest.skip("https://github.com/nod-ai/amdshark-ai/issues/44")
    def testTorchImplNonTransposedRHS(self):
        ops._registry._test_enable_last_op_dispatch(True)
        t1 = torch.rand(32, 16, dtype=torch.float32)
        t2 = torch.rand(16, 48, dtype=torch.float16)
        result = ops.matmul(t1, t2)
        expected = torch.matmul(t1, t2.to(torch.float32))
        assert_tensor_close(result, expected)
        self.assertIsNot(
            ops._registry._test_get_last_op_dispatch(),
            ops.custom_impls.matmul_mmtfp_tensor_tensor,
        )

    @unittest.skip("https://github.com/nod-ai/amdshark-ai/issues/44")
    def testTorchImplTransposedPrimitiveRHS(self):
        ops._registry._test_enable_last_op_dispatch(True)
        t1 = torch.rand(32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        t2_pt = DefaultPrimitiveTensor(data=t2)
        result = ops.matmul(t1, t2_pt.T)
        expected = torch.matmul(t1, t2.T.to(torch.float32))
        assert_tensor_close(result, expected)
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.custom_impls.matmul_mmtfp_tensor_tensor,
        )

    def testTorchImplImplicitBatch(self):
        ops._registry._test_enable_last_op_dispatch(True)
        t1 = torch.rand(4, 32, 16, dtype=torch.float32)
        t2 = torch.rand(48, 16, dtype=torch.float16)
        t2_pt = DefaultPrimitiveTensor(data=t2)
        result = ops.matmul(t1, t2_pt.T)
        expected = torch.matmul(t1, t2.T.to(torch.float32))
        assert_tensor_close(result, expected)

    def testTorchImplTransposedQuantizedRHS_BlockScaledLayout(self):
        ops._registry._test_enable_last_op_dispatch(True)
        a_dtype = torch.float32
        d_dtype = torch.float32
        ref_dtype = torch.float32
        a = torch.rand([4, 16, 3200], dtype=a_dtype) * 64
        d = torch.rand([3200, 100, 1], dtype=d_dtype) * 64
        qs = (torch.rand([3200, 100, 32], dtype=ref_dtype) * 32.0).to(torch.int8)
        rhs_pqt = PlanarQuantizedTensor(
            shape=[3200, 3200], layout=BlockScaledLayout([3200, 3200], d, qs)
        )
        result = ops.matmul(a, rhs_pqt, transpose_rhs=True)
        # Just verifying dispatch. Numerics are tested at the kernel level.
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.custom_impls.matmul_generic_tensor_block_scaled,
        )

    def testTorchImplTransposedQuantizedRHS_BlockScaledOffsetI4(self):
        ops._registry._test_enable_last_op_dispatch(True)
        a_dtype = torch.float32
        d_dtype = torch.float32
        ref_dtype = torch.float32
        a = torch.rand([4, 16, 3200], dtype=a_dtype) / 256.0
        d = torch.rand([3200, 100, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([3200, 100, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([3200, 100, 1], dtype=d_dtype) + 16.0
        rhs_pqt = PlanarQuantizedTensor(
            shape=[3200, 3200],
            layout=BlockScaledI4Layout([3200, 3200], d, qs, m=m, signed=False),
        )
        result = ops.matmul(a, rhs_pqt, transpose_rhs=True)
        # Just verifying dispatch. Numerics are tested at the kernel level.
        self.assertIs(
            ops._registry._test_get_last_op_dispatch(),
            ops.custom_impls.matmul_generic_tensor_block_scaled_i4,
        )

    # TODO: mmt_super_block_scaled_offset_q4_unsigned


class OnesTest(unittest.TestCase):
    def testEndOnly(self):
        expected = torch.ones(5)
        actual = ops.ones(5)
        assert ops.equal(expected, actual)

    def testShape(self):
        expected = torch.ones(3, 2)
        actual = ops.ones(3, 2)
        assert ops.equal(expected, actual)

    def testTupleShape(self):
        expected = torch.ones((3, 2, 1))
        actual = ops.ones((3, 2, 1))
        assert ops.equal(expected, actual)

    @parameterized.expand([torch.float32, torch.int64])
    def testDtype(self, dtype):
        expected = torch.ones(5, dtype=dtype)
        actual = ops.ones(5, dtype=dtype)
        assert expected.dtype == actual.dtype
        assert ops.equal(expected, actual)


@pytest.mark.usefixtures("iree_flags")
class IndexCopyTest(unittest.TestCase):
    @parameterized.expand([torch.float8_e4m3fnuz, torch.float16])
    def testEagerVsIREE(self, dtype: torch.dtype):
        class Module(torch.nn.Module):
            def forward(
                self, inout: torch.Tensor, index: torch.Tensor, tensor: torch.Tensor
            ) -> torch.Tensor:
                return ops.index_copy_(inout, 0, index, tensor)

        x = torch.zeros(5, 3, dtype=dtype)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        index = torch.tensor([0, 4, 2])

        module = Module()
        expected_x = x.clone()
        module(expected_x, index, t)

        actual_x = x.clone()
        oneshot_iree_run(
            module,
            args=(actual_x, index, t),
            compile_args=get_iree_compiler_flags_from_object(self),
            device=self.iree_device,
        )
        assert_tensor_close(actual_x, expected_x, atol=0, rtol=0)


@pytest.mark.usefixtures("iree_flags")
class IndexPutTest(unittest.TestCase):
    @parameterized.expand([torch.float8_e4m3fnuz, torch.float16])
    def testEagerVsIREE(self, dtype: torch.dtype):
        class Module(torch.nn.Module):
            def forward(
                self, inout: torch.Tensor, index: torch.Tensor, tensor: torch.Tensor
            ) -> torch.Tensor:
                return ops.index_put_(inout, (index,), tensor)

        x = torch.zeros(5, 3, dtype=dtype)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        index = torch.tensor([0, 4, 2])

        module = Module()
        expected_x = x.clone()
        module(expected_x, index, t)

        actual_x = x.clone()
        oneshot_iree_run(
            module,
            args=(actual_x, index, t),
            compile_args=get_iree_compiler_flags_from_object(self),
            device=self.iree_device,
        )
        assert_tensor_close(actual_x, expected_x, atol=0, rtol=0)


class InvertTest(unittest.TestCase):
    def testInvertPrimitiveTensor(self):
        tensor = torch.rand(2, 3).to(torch.bool)
        expected_result = ~tensor
        actual_result = ~DefaultPrimitiveTensor(data=tensor)
        assert ops.equal(actual_result, expected_result)

    def testInvertReplicatedTensor(self):
        tensor = torch.rand(2, 3).to(torch.bool)
        expected_result = ~tensor
        actual_result = ~ReplicatedTensor(ts=tensor, shard_count=2)
        assert ops.equal(actual_result, expected_result)

    def testInvertSplitTensor(self):
        tensor = torch.rand(2, 3).to(torch.bool)
        expected_result = ~tensor
        actual_result = ~SplitPrimitiveTensor(ts=tensor, shard_dim=0, shard_count=2)
        assert ops.equal(actual_result, expected_result)


class PermuteTest(unittest.TestCase):
    def testPermute(self):
        torch_tensor = torch.rand(3, 4, 5, dtype=torch.float32)
        permutation = [1, 0, 2]
        primitive_tensor = DefaultPrimitiveTensor(data=torch_tensor)
        expected_result = torch.permute(torch_tensor, permutation)

        permuted_torch_tensor = ops.permute(torch_tensor, permutation)
        permuted_primitive_tensor = ops.permute(primitive_tensor, permutation)

        assert ops.equal(expected_result, permuted_torch_tensor)
        assert ops.equal(expected_result, permuted_primitive_tensor)

    def testTensorPropertyT(self):
        torch_tensor = torch.rand(3, 5, dtype=torch.float32)
        primitive_tensor = DefaultPrimitiveTensor(data=torch_tensor)
        assert ops.equal(torch_tensor.T, primitive_tensor.T)


class RmsNormTest(unittest.TestCase):
    def _ref(self, x, weight, epsilon):
        variance = x.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + epsilon)
        output = output * weight
        return output

    def testTorchImpl(self):
        t1 = torch.rand(16, 128, dtype=torch.float32)
        t2 = torch.rand(16, 128, dtype=torch.float32)
        result = ops.rms_norm(t1, t2, epsilon=1e-10, orig_dtype=torch.float32)
        actual = self._ref(t1, t2, epsilon=1e-10)
        assert_tensor_close(actual, result)

    def testTorchPrimitiveWeightImpl(self):
        t1 = torch.rand(16, 128, dtype=torch.float32)
        t2 = torch.rand(16, 128, dtype=torch.float32)
        t2_pt = DefaultPrimitiveTensor(data=t2)
        result = ops.rms_norm(t1, t2_pt, epsilon=1e-10, orig_dtype=torch.float32)
        actual = self._ref(t1, t2, epsilon=1e-10)
        assert_tensor_close(actual, result)

    # TODO: Quantized tensor


class TransferAndBarrierTest(TempDirTestBase):
    class Module(BaseLayer):
        def __init__(
            self, target_devices: list[int], op: Callable[[AnyTensor, int], AnyTensor]
        ):
            super().__init__()
            self.target_devices = target_devices
            self.op = op

        def forward(self, x: AnyTensor):
            if isinstance(x, ShardedTensor):
                assert len(self.target_devices) == len(
                    x.shards
                ), "Should have received one target device for each shard."
                shards = [
                    self.op(shard, target_device)
                    for shard, target_device in zip(x.shards, self.target_devices)
                ]
                return x.clone(ts=shards)

            assert (
                len(self.target_devices) == 1
            ), "Should have only received one target device for unsharded tensor."
            return self.op(x, self.target_devices[0])

    op_to_mlir_name = {
        ops.transfer_to_logical_device: "flow.tensor.transfer",
        ops.barrier_on_logical_device: "flow.tensor.barrier",
    }

    def setUp(self):
        super().setUp()
        self.mlir_path = self._temp_dir / "model.mlir"

    def look_for_op(self, op: Callable, expected_count: int):
        """
        Search through the provided MLIR file and find the specified operation.
        Will throw and error if the operation is not found or if the count does not match.

        Args:
            op: The op to search the MLIR for.
            expected_count: Expected number of occurrences of the operation.
        """
        with open(self.mlir_path, "r") as f:
            mlir_contents = f.read()

        op_name = self.op_to_mlir_name[op]
        transfer_per_target_device = {i: 0 for i in self.device_ordinals}
        for line in mlir_contents.splitlines():
            if op_name in line:
                target_device_ordinal = int(
                    line.split("#hal.device.promise<@__device_")[1].split(">")[0]
                )
                transfer_per_target_device[target_device_ordinal] += 1

        for device_ordinal, actual_count in transfer_per_target_device.items():
            assert (
                actual_count == expected_count
            ), f"Expected {expected_count} transfers to device {device_ordinal}, but found {actual_count}."

    def run_test(
        self,
        *,
        tensor_class: torch.Tensor.__class__ | InferenceTensor.__class__,
        base_tensor: AnyTensor | None,
        op: Callable,
    ):
        self.shard_count = 2 if issubclass(tensor_class, ShardedTensor) else 1
        tensor = create_sample_tensor_from_class(
            tensor_class, shard_count=self.shard_count, base_tensor=base_tensor
        )
        subtensor_dict = (
            {0: tensor} if isinstance(tensor, torch.Tensor) else tensor.subtensors
        )
        self.device_ordinals = list(1 + 2 * i for i in range(self.shard_count))

        model = self.Module(target_devices=self.device_ordinals, op=op)
        fxb = FxProgramsBuilder(model)

        @fxb.export_program(name="forward", args=(subtensor_dict,), strict=False)
        def _(model, subtensors: list[torch.Tensor]):
            reconstructed_tensor = (
                subtensors[0]
                if isinstance(tensor, torch.Tensor)
                else tensor._clone_with_subtensors(subtensor_dict)
            )

            output_tensor = model(reconstructed_tensor)

            return (
                list(output_tensor.subtensors.values())
                if isinstance(output_tensor, InferenceTensor)
                else [output_tensor]
            )

        output = aot.export(fxb)
        output.save_mlir(self.mlir_path)

        # 3. Look for transfer op
        expected_count = 1
        if isinstance(tensor, InferenceTensor):
            expected_count = len(tensor.subtensors) // self.shard_count
        self.look_for_op(op, expected_count)

    @parameterized.expand(
        [
            (op, tensor_type)
            for op, tensor_type in product(
                [
                    ops.transfer_to_logical_device,
                    ops.barrier_on_logical_device,
                ],
                [
                    torch.Tensor,
                    DefaultPrimitiveTensor,
                    BlockScaledFp4Layout,
                    BlockScaledI4Layout,
                    SuperBlockOffsetScaled_4_6_Layout,
                ],
            )
        ]
    )
    def testBarrierTransferUnsharded(
        self,
        op: Callable[[AnyTensor, int], AnyTensor],
        tensor_class: torch.Tensor.__class__ | InferenceTensor.__class__,
    ):
        self.run_test(tensor_class=tensor_class, base_tensor=None, op=op)

    @parameterized.expand(
        [
            (op, tensor_type, shard_type)
            for op, tensor_type, shard_type in product(
                [
                    ops.transfer_to_logical_device,
                    ops.barrier_on_logical_device,
                ],
                [
                    ReplicatedTensor,
                    SplitPrimitiveTensor,
                    UnreducedTensor,
                ],
                [
                    torch.Tensor,
                    DefaultPrimitiveTensor,
                    BlockScaledFp4Layout,
                    BlockScaledI4Layout,
                    SuperBlockOffsetScaled_4_6_Layout,
                ],
            )
        ]
    )
    def testBarrierTransferSharded(
        self,
        op: Callable[[AnyTensor, int], AnyTensor],
        tensor_class: torch.Tensor.__class__ | InferenceTensor.__class__,
        shard_type: torch.Tensor.__class__ | InferenceTensor.__class__,
    ):
        base_tensor = create_sample_tensor_from_class(shard_type)
        self.run_test(tensor_class=tensor_class, base_tensor=base_tensor, op=op)


class TestOpExport(unittest.TestCase):
    """Tests that the machinery holds up under dynamo torch.export.

    Dynamo can be finicky with dynamism, and we've had trouble, so verify.
    """

    def testExport(self):
        a_dtype = torch.float32
        d_dtype = torch.float32
        ref_dtype = torch.float32
        a = torch.rand([4, 16, 3200], dtype=a_dtype) / 256.0
        d = torch.rand([3200, 100, 1], dtype=d_dtype) / 256.0
        qs = (torch.rand([3200, 100, 16], dtype=ref_dtype) * 255.0).to(torch.uint8)
        m = torch.rand([3200, 100, 1], dtype=d_dtype) + 16.0

        class MyModule(torch.nn.Module):
            def forward(self, a, d, qs, m):
                rhs_pqt = PlanarQuantizedTensor(
                    shape=[3200, 3200],
                    layout=BlockScaledI4Layout([3200, 3200], d, qs, m=m, signed=False),
                )
                result = ops.linear(a, rhs_pqt)
                return result

        my_module = MyModule()
        ep = torch.export.export(my_module, (a, d, qs, m))
        s = str(ep)
        self.assertIn("mmt_block_scaled_offset_q4_unsigned.default", s)


class TestScatter(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)
        self.rng = np.random.default_rng(0)

    def testInplaceSourceAsNumber(self):
        dim = 1
        input = torch.randint(low=0, high=10, size=[3, 8, 5], dtype=torch.int32)
        index = torch.tensor(
            self.rng.choice(input.shape[dim], size=[2, 2, 2], replace=False)
        )
        src = 2
        expected = input.scatter_(dim, index, src)
        actual = DefaultPrimitiveTensor(data=input).scatter_(dim, index, src)
        assert ops.equal(actual, expected)

    def testInplaceSourceAsTensor(self):
        dim = 1
        input = torch.randint(low=0, high=10, size=[3, 8, 5], dtype=torch.int32)
        index = torch.tensor(
            self.rng.choice(input.shape[dim], size=[2, 2, 2], replace=False)
        )
        src = torch.randint_like(index, low=0, high=10, dtype=torch.int32)
        expected = input.scatter_(dim, index, src)
        actual = DefaultPrimitiveTensor(data=input).scatter_(dim, index, src)
        assert ops.equal(actual, expected)


class TestScatterAdd(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)

    def test(self):
        dim = 1
        input = torch.randint(low=0, high=10, size=[3, 4, 5], dtype=torch.int32)
        index = torch.randint(
            low=0, high=input.shape[dim], size=[3, 10, 5], dtype=torch.int64
        )
        src = torch.randint_like(index, low=0, high=10, dtype=torch.int32)
        expected = input.scatter_add(dim, index, src)
        actual = DefaultPrimitiveTensor(data=input).scatter_add(dim, index, src)
        assert ops.equal(actual, expected)


class TensorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    @parameterized.expand(product([True, 1, 1.0, 1 + 2j], [None, (1,), (1, 2)]))
    def testScalarInput(
        self,
        data: Number,
        devices: tuple[int, ...] | None,
    ):
        expected = torch.tensor(data)
        actual = ops.tensor(data, devices=devices)
        assert ops.equal(actual, expected)
        if devices is not None:
            assert isinstance(actual, ReplicatedTensor)
            assert tuple(devices) == actual.devices
            assert actual.shard_count == len(devices)
            assert all(ops.equal(shard, expected) for shard in actual.shards)

    @parameterized.expand([None, 1, 2])
    def testTensorInputWithoutDevices(self, shard_count: int | None):
        input = torch.randn(3, 4, dtype=torch.float32)
        expected = torch.tensor(input)

        if shard_count:
            input = ReplicatedTensor(ts=input, shard_count=shard_count)
        actual = ops.tensor(input)
        assert actual.dtype == expected.dtype == input.dtype
        assert ops.equal(actual, expected)
        if shard_count:
            assert isinstance(actual, ReplicatedTensor)
            assert actual.shard_count == shard_count
            assert actual.devices == tuple(range(shard_count))
            assert all(ops.equal(shard, expected) for shard in actual.shards)


class TestTopK(unittest.TestCase):
    @parameterized.expand(
        [
            (-1, 4, True, True, (1, 1, 256), 16, False),
            (-1, 4, True, True, (1, 1, 256), 8, False),
            (-1, 8, True, True, (1, 1, 256), 16, False),
            (-1, 4, False, True, (1, 1, 256), 16, False),
            (-1, 4, False, False, (1, 1, 256), 16, False),
            (-1, 2, True, True, (2, 1, 6), 3, False),
            (-1, 4, True, False, (1, 1, 64), 8, True),
        ]
    )
    def testSplitTopKLastDim(
        self, dim, k, largest, _sorted, shape, chunk_size, use_linalgext_topk
    ):
        numels = math.prod(shape)
        tensor = torch.arange(numels) * 173 + 129
        tensor = tensor % numels
        tensor = tensor.to(torch.float16).view(shape)

        values_expected, index_expected = torch.topk(tensor, k, dim, largest, _sorted)

        values, index = ops.topk(
            tensor,
            k,
            dim,
            largest,
            _sorted,
            chunk_size=chunk_size,
            use_linalgext_topk=use_linalgext_topk,
        )

        if _sorted is False:
            values = torch.sort(values).values
            values_expected = torch.sort(values_expected).values

        assert_tensor_close(values, values_expected)
        index = index.to(torch.int64)

        values_from_indices = torch.gather(tensor, -1, index=index)
        values_from_indices_expected = torch.gather(tensor, -1, index=index_expected)

        if _sorted is False:
            values_from_indices = torch.sort(values_from_indices).values
            values_from_indices_expected = torch.sort(
                values_from_indices_expected
            ).values

        assert_tensor_close(values_from_indices, values_from_indices_expected)

    @parameterized.expand(
        [
            (0, 2, True, True, (4, 1, 8), 2),
            (0, 2, False, True, (4, 1, 8), 2),
            (0, 2, False, False, (4, 1, 8), 2),
            (0, 2, True, False, (4, 1, 8), 2),
            (0, 3, True, True, (6, 2, 12), 3),
            (0, 4, True, True, (8, 3, 24), 4),
        ]
    )
    def testSplitTopKDim0(self, dim, k, largest, _sorted, shape, chunk_size):
        numels = math.prod(shape)
        tensor = torch.arange(numels) * 173 + 129
        tensor = tensor % numels
        tensor = tensor.to(torch.float16).view(shape)

        values_expected, index_expected = torch.topk(tensor, k, dim, largest, _sorted)

        values, index = ops.topk(
            tensor, k, dim, largest, _sorted, chunk_size=chunk_size
        )

        if not _sorted:
            values = torch.sort(values, dim=dim).values
            values_expected = torch.sort(values_expected, dim=dim).values

        assert_tensor_close(values, values_expected)

        # Duplicate values may cause differences in indices
        index_slices = [slice(None)] * tensor.ndim
        index_slices[dim] = index[0, 0]
        values_from_indices = tensor[tuple(index_slices)]

        index_slices_expected = [slice(None)] * tensor.ndim
        index_slices_expected[dim] = index_expected[0, 0]
        values_from_indices_expected = tensor[tuple(index_slices_expected)]

        if not _sorted:
            values_from_indices = torch.sort(values_from_indices, dim=dim).values
            values_from_indices_expected = torch.sort(
                values_from_indices_expected, dim=dim
            ).values

        assert_tensor_close(values_from_indices, values_from_indices_expected)


class TestTraceTensors(TempDirTestBase):
    def setUp(self):
        super().setUp()
        self.callback_stash = debugging.get_trace_tensor_callback()
        debugging.set_trace_tensor_callback(
            debugging.trace_tensor_to_safetensors_callback
        )

        self.enable_tensor_trace_stash = debugging.flags.enable_tensor_trace
        debugging.flags.enable_tensor_trace = True

        self.trace_path_stash = debugging.flags.trace_path
        debugging.flags.trace_path = self._temp_dir

    def tearDown(self):
        super().tearDown()
        debugging.set_trace_tensor_callback(self.callback_stash)
        debugging.flags.enable_tensor_trace = self.enable_tensor_trace_stash
        debugging.flags.trace_path = self.trace_path_stash

    def testTraceOneTensorInEagerMode(self):
        tensor = torch.arange(1, 5)
        trace_key = "test_trace_key"
        ops.trace_tensor(trace_key, tensor)

        trace_filepath = debugging.flags.trace_path / f"{trace_key}.safetensors"
        with safetensors.safe_open(trace_filepath, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
        assert_tensor_close(recorded_tensor, tensor, rtol=0, atol=0)

    def testTraceOneShardedTensorInEagerMode(self):
        tensor = torch.arange(1, 6)
        sharded_tensor = ops.reshard_split(tensor, count=2, dim=0)
        trace_key = "test_trace_key"
        ops.trace_tensor(trace_key, sharded_tensor)

        trace_filepath = debugging.flags.trace_path / f"{trace_key}.safetensors"
        with safetensors.safe_open(trace_filepath, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
        assert_tensor_close(recorded_tensor, tensor, rtol=0, atol=0)

    def testTraceTensorWithIree(self):
        trace_key = "test_trace_key"
        tensor = torch.arange(1, 6, dtype=torch.float32)

        class Module(BaseLayer):
            def forward(self, x: torch.Tensor):
                self.trace_tensor(trace_key, x)
                return x

        model = Module()
        fxb = FxProgramsBuilder(model)

        @fxb.export_program(
            name="forward",
            args=(tensor,),
            strict=False,
        )
        def _(model, x):
            return model(x)

        output = aot.export(fxb)
        mlir_path = self._temp_dir / "model.mlir"
        output.save_mlir(mlir_path)
        iree_module_path = self._temp_dir / "model.vmfb"
        iree.compiler.compile_file(
            str(mlir_path),
            output_file=str(iree_module_path),
            extra_args=[
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-hal-target-device=local",
            ],
        )

        iree_devices = get_iree_devices(driver="local-task", device_count=1)

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            iree_buffere_view_trace_callback = (
                make_hal_buffer_view_trace_default_callback(iree_devices[0])
            )
            debug_sink = iree.runtime.HalModuleDebugSink(
                iree_buffere_view_trace_callback
            )
            iree_module, iree_vm_context, _ = load_iree_module(
                module_path=str(iree_module_path),
                devices=iree_devices,
                debug_sink=debug_sink,
            )
            iree_args = prepare_iree_module_function_args(
                args=[tensor],
                devices=iree_devices,
            )
            run_iree_module_function(
                module=iree_module,
                vm_context=iree_vm_context,
                args=iree_args,
                device=iree_devices[0],
                function_name=f"forward",
            )

        with_iree_device_context(run_iree_module, iree_devices)

        trace_filepath = debugging.flags.trace_path / f"{trace_key}.safetensors"
        with safetensors.safe_open(trace_filepath, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
        assert_tensor_close(recorded_tensor, tensor, rtol=0, atol=0)

    def testTraceInNestedModules(self):
        tensor = torch.arange(1, 6)
        trace_key = "test_trace_key"

        class ModuleC(BaseLayer):
            def forward(self):
                self.trace_tensor(trace_key, {"the_tensor": tensor})
                return

        class ModuleB(BaseLayer):
            def __init__(self):
                super().__init__()
                self.c = ModuleC()

            def forward(self):
                return self.c()

        class ModuleA(BaseLayer):
            def __init__(self):
                super().__init__()
                self.b = ModuleB()

            def forward(self):
                return self.b()

        a = ModuleA()
        a.set_recursively_submodules_default_trace_tensor_key_prefix()

        a()
        trace_filepath = (
            debugging.flags.trace_path / f"b.c.{trace_key}.the_tensor.safetensors"
        )
        with safetensors.safe_open(trace_filepath, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 1
            recorded_tensor = f.get_tensor("")
        assert_tensor_close(recorded_tensor, tensor, rtol=0, atol=0)


class TransposeTest(unittest.TestCase):
    def testPrimitiveTensor(self):
        tensor = torch.tensor([[1, 2], [3, 4]])
        expected_transposed = torch.transpose(tensor, 0, 1)

        transposed_tensor = DefaultPrimitiveTensor(data=tensor).transpose(0, 1)
        assert isinstance(transposed_tensor, DefaultPrimitiveTensor)
        retransposed_tensor = transposed_tensor.transpose(0, 1)
        assert isinstance(retransposed_tensor, DefaultPrimitiveTensor)

        assert ops.equal(expected_transposed, transposed_tensor)
        assert ops.equal(tensor, retransposed_tensor)

    def quantized_tensor_helper(
        self, quantizer: QuantizerTensor, expected: torch.Tensor
    ):
        expected_transposed = expected.transpose(0, 1)

        quantized = quantizer.quantize(expected)
        transposed_quantized = quantized.transpose(0, 1)
        retransposed_quantized = transposed_quantized.transpose(0, 1)

        dequantized = quantized.layout.dequant()
        assert ops.equal(expected, dequantized)

        dequantized_transposed = transposed_quantized.layout.dequant()
        assert ops.equal(expected_transposed, dequantized_transposed)

        dequantized_retransposed = retransposed_quantized.layout.dequant()
        assert ops.equal(expected, dequantized_retransposed)

    def testTensorScaled(self):
        expected = torch.tensor([[-6, -4, -2, 0], [-6, -4, -2, 0]], dtype=torch.float32)
        quantizer = StaticScaledQuantizer(
            scale=torch.tensor(0.5, dtype=torch.float32),
            offset=torch.tensor(5.0, dtype=torch.float32),
            dtype=torch.float32,
        )
        self.quantized_tensor_helper(quantizer, expected)

    def testBlockScaledFp4(self):
        expected = torch.tensor(
            [[[-6, -4, -2, 0], [-4, -3, -2, -1]], [[6, 4, 2, 1], [4, 3, 1, -1]]],
            dtype=torch.float32,
        )
        block_size = 2
        scales_shape = list(expected.shape) + [1]
        scales_shape[-2] //= block_size
        quantizer = StaticFp4BlockQuantizer(
            scales=torch.ones(size=scales_shape, dtype=torch.float32),
            dtype=torch.float32,
            block_size=block_size,
            use_fe8m0_scale=False,
        )
        self.quantized_tensor_helper(quantizer, expected)

    def testBlockScaledFp4ShouldFail(self):
        expected = torch.tensor(
            [[-6, -4, -2, 0], [-5, -3, -2, -1]], dtype=torch.float32
        )
        block_size = 2
        scales_shape = list(expected.shape) + [1]
        scales_shape[-2] //= block_size
        quantizer = StaticFp4BlockQuantizer(
            scales=torch.full(size=scales_shape, fill_value=0.5, dtype=torch.float32),
            dtype=torch.float32,
            block_size=block_size,
            use_fe8m0_scale=False,
        )
        with pytest.raises(
            ValueError, match="Cannot transpose last dim of BlockScaledLayout tensors."
        ):
            self.quantized_tensor_helper(quantizer, expected)


class ConvTest(unittest.TestCase):
    def testConv2d(self):
        # Random input tensor: batch size = 1, channels = 1, height = 5, width = 5
        input = torch.rand(1, 1, 5, 5)
        # Random kernel: out_channels = 1, in_channels = 1, kernel_size = 3x3
        weight = torch.rand(1, 1, 3, 3)
        result = ops.conv2d(input, weight)
        expected = torch.conv2d(input, weight)
        assert_tensor_close(result, expected)

    def testConv3d(self):
        # Random input tensor: batch size = 1, channels = 1, depth = 4, height = 4, width = 4
        input = torch.rand(1, 1, 4, 4, 4)
        # Random kernel: out_channels = 1, in_channels = 1, kernel_size = 2x2x2
        weight = torch.rand(1, 1, 2, 2, 2)
        result = ops.conv3d(input, weight)
        expected = torch.conv3d(input, weight)
        assert_tensor_close(result, expected)

    def testConv1d(self):
        # Random input tensor: batch size = 1, channels = 1, width = 10
        input = torch.rand(1, 1, 10)
        # Random kernel: out_channels = 1, in_channels = 1, kernel_size = 3
        weight = torch.rand(1, 1, 3)
        result = ops.conv1d(input, weight)
        expected = torch.conv1d(input, weight)
        assert_tensor_close(result, expected)


class SwigluTest(unittest.TestCase):
    def _ref(
        self, x: torch.Tensor, *, alpha: float = 1.702, limit: float | None = None
    ):
        # Reference matches the default_impls logic:
        # x_glu = x[..., 0::2]; x_lin = x[..., 1::2]
        x_glu = x[..., ::2]
        x_lin = x[..., 1::2]

        if limit is not None:
            x_glu = x_glu.clamp(min=None, max=limit)
            x_lin = x_lin.clamp(min=-limit, max=limit)

        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        return out_glu * (x_lin + 1)

    @parameterized.expand(
        [
            ((2, 3, 8), torch.float16, 1.702, None),
            ((2, 3, 8), torch.float32, 1.702, None),
            ((4, 5, 16), torch.float16, 1.5, None),
            ((4, 5, 16), torch.float32, 1.5, 6.0),
            ((1, 1, 32), torch.float16, 1.702, 5.0),
            ((1, 1, 32), torch.float32, 2.0, 7.0),
        ]
    )
    def testSwiGLUMatchesReference(self, shape, dtype, alpha, limit):
        torch.random.manual_seed(0)
        x = torch.randn(*shape, dtype=dtype)
        expected = self._ref(x, alpha=alpha, limit=limit)
        actual = ops.swiglu(x, alpha=alpha, limit=limit)
        assert_tensor_close(actual, expected)

    def testSwiGLURaisesOnOddLastDim(self):
        x = torch.randn(2, 3, 7, dtype=torch.float32)  # last dim is odd
        with pytest.raises(ValueError, match="SwiGLU expects even last dim"):
            _ = ops.swiglu(x)


@parameterized_class("shard_count", [(None,), (1,), (2,)])
class WhereTest(unittest.TestCase):
    def make_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype | None = None
    ) -> AnyTensor:
        tensor = torch.randn(shape, dtype=dtype)
        if self.shard_count is None:
            return tensor
        return ReplicatedTensor(ts=tensor, shard_count=self.shard_count)

    def test_where_condition_only(self):
        x = self.make_tensor((3, 2))
        condition = x > 0
        expected = torch.where(unbox_tensor(condition))
        actual = ops.where(condition)

        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), len(expected))
        for actual_i, expected_i in zip(actual, expected):
            assert_tensor_close(actual_i, expected_i)

    def test_where_selection_input_other_scalar(self):
        x = self.make_tensor((3, 2))
        condition = x > 0
        expected = torch.where(unbox_tensor(condition), 1.0, 0.0)
        actual = ops.where(condition, 1.0, 0.0)
        assert_tensor_close(actual, expected)

    def test_where_selection_input_other_tensor(self):
        x = self.make_tensor((3, 2))
        y = self.make_tensor((3, 2))
        condition = x > 0
        expected = torch.where(
            unbox_tensor(condition), unbox_tensor(x), unbox_tensor(y)
        )
        actual = ops.where(condition, x, y)
        assert_tensor_close(actual, expected)

    def test_where_selection_input_tensor_other_scalar(self):
        x = self.make_tensor((3, 2))
        y = 1.0
        condition = x > 0
        expected = torch.where(unbox_tensor(condition), unbox_tensor(x), y)
        actual = ops.where(condition, x, y)
        assert_tensor_close(actual, expected)

    def test_where_selection_input_scalar_other_tensor(self):
        x = 1.0
        y = self.make_tensor((3, 2))
        condition = y > 0
        expected = torch.where(unbox_tensor(condition), x, unbox_tensor(y))
        actual = ops.where(condition, x, y)
        assert_tensor_close(actual, expected)

    @parameterized.expand([torch.float16, torch.float32, torch.float64])
    def test_where_dtype_preservation(self, dtype: torch.dtype):
        x = self.make_tensor((2, 2), dtype=dtype)
        condition = x > 0
        expected = torch.where(unbox_tensor(condition), unbox_tensor(x), 0)
        actual = ops.where(condition, x, 0)
        assert_tensor_close(actual, expected)
        self.assertEqual(actual.dtype, dtype)

    def test_where_broadcasting(self):
        condition = self.make_tensor((2, 2)) > 0
        x = self.make_tensor((2, 1))
        y = self.make_tensor((1, 2))
        expected = torch.where(
            unbox_tensor(condition), unbox_tensor(x), unbox_tensor(y)
        )
        actual = ops.where(condition, x, y)
        assert_tensor_close(actual, expected)


class ZerosTest(unittest.TestCase):
    def testEndOnly(self):
        expected = torch.zeros(5)
        actual = ops.zeros(5)
        assert ops.equal(expected, actual)

    def testShape(self):
        expected = torch.zeros(3, 2)
        actual = ops.zeros(3, 2)
        assert ops.equal(expected, actual)

    def testTupleShape(self):
        expected = torch.zeros((3, 2, 1))
        actual = ops.zeros((3, 2, 1))
        assert ops.equal(expected, actual)

    @parameterized.expand([torch.float32, torch.int64])
    def testDtype(self, dtype):
        expected = torch.zeros(5, dtype=dtype)
        actual = ops.zeros(5, dtype=dtype)
        assert expected.dtype == actual.dtype
        assert ops.equal(expected, actual)

    @parameterized.expand(
        [
            ((3,),),
            ((3, 2),),
            ((3, 2, 1),),
        ]
    )
    def testDevicesReplicated(self, devices: tuple[int, ...]):
        expected = torch.zeros(5, dtype=torch.float32)
        actual = ops.zeros(5, devices=devices, dtype=torch.float32)
        assert isinstance(actual, ReplicatedTensor)
        assert tuple(devices) == actual.devices
        assert actual.shard_count == len(devices)
        assert all(ops.equal(shard, expected) for shard in actual.shards)


if __name__ == "__main__":
    unittest.main()
