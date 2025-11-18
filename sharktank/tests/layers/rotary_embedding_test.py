# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import math
import pytest
import torch

import iree.runtime
import iree.turbine.aot as aot
from iree.compiler import compile_file

from parameterized import parameterized_class

from amdsharktank import ops
from amdsharktank.layers import CachedRotaryLayer, build_rotary_layer
from amdsharktank.types import AnyTensor, ReplicatedTensor
from amdsharktank.utils.iree import device_array_to_host, tensor_to_device_array
from amdsharktank.utils.testing import TempDirTestBase, assert_tensor_close


def validate(
    xq: AnyTensor,
    em: AnyTensor,
    rope_dims: float,
    rope_freq_base: float,
    interleaved: bool,
) -> None:
    xq = ops.unshard(xq)
    em = ops.unshard(em)

    # Initially we want to compute the lengths of each vector
    if interleaved:
        xq_01 = xq.unflatten(-1, (rope_dims // 2, 2))
        em_01 = em.unflatten(-1, (2, rope_dims // 2))
        em_01 = em_01.transpose(-2, -1)
    else:
        xq_01 = xq.unflatten(-1, (2, rope_dims // 2))
        em_01 = em.unflatten(-1, (2, rope_dims // 2))
        xq_01 = xq_01.transpose(-2, -1)
        em_01 = em_01.transpose(-2, -1)

    xq_0 = xq_01[:, :, :, :, 0]
    xq_1 = xq_01[:, :, :, :, 1]

    em_0 = em_01[:, :, :, :, 0]
    em_1 = em_01[:, :, :, :, 1]

    xq_l = torch.sqrt(xq_0 * xq_0 + xq_1 * xq_1)
    em_l = torch.sqrt(em_0 * em_0 + em_1 * em_1)
    assert_tensor_close(xq_l, em_l)

    # Normalize
    xq_0 = xq_0 / xq_l
    xq_1 = xq_1 / xq_l
    em_0 = em_0 / em_l
    em_1 = em_1 / em_l

    # Compute the angle step per value
    xq_a = torch.atan2(xq_1, xq_0)
    em_a = torch.atan2(em_1, em_0)

    # Compute the step size for the rotation
    angle = em_a - xq_a
    angle = angle[:, 1:, :, :] - angle[:, :-1, :, :]
    step = angle[0, 1, 0, :][None, None, None, :]
    step = ops.where(step > math.pi * 2.0, step - math.pi * 2.0, step)
    step = ops.where(step < 0.0, step + math.pi * 2.0, step)

    # Check that the step size is approximately correct
    expected_step = ops.log(torch.tensor(rope_freq_base)) * (
        -(ops.arange(rope_dims // 2)) / (rope_dims // 2)
    )
    expected_step = torch.exp(expected_step)
    assert_tensor_close(step.flatten(), expected_step, atol=1e-2, rtol=1e-2)

    # Guarantee a progressive stepping for rotation:
    angle = angle / step
    angle = angle[:, 1:, ::]
    angle = ops.where(angle < 0, angle + math.pi * 2.0, angle)
    assert_tensor_close(angle, torch.full(angle.shape, 1.0), atol=1e-2, rtol=1e-2)


@pytest.mark.usefixtures("iree_flags")
@parameterized_class(
    ("interleave_rotary",),
    [(True,), (False,)],
)
class TestRotaryEmbedding(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.bs = 1
        self.rope_dims = 8
        self.heads = 1
        self.max_seqlen = 16
        self.rope_freq_base = 10000.0
        self.pipeline_stage_to_device_map: list[list[int]] | None = None

        self.xq = torch.rand(
            (self.bs, self.max_seqlen, self.heads, self.rope_dims),
            dtype=torch.float,
        )

    def validate(self, em: AnyTensor, xq: AnyTensor) -> None:
        if isinstance(em, ReplicatedTensor):
            assert em.devices == xq.devices

        validate(
            xq=self.xq,
            em=em,
            rope_dims=self.rope_dims,
            rope_freq_base=self.rope_freq_base,
            interleaved=self.interleave_rotary,
        )

    def create_rotary_layer(self) -> CachedRotaryLayer:
        return build_rotary_layer(
            rope_dimension_count=self.rope_dims,
            rope_freq_base=self.rope_freq_base,
            interleave=self.interleave_rotary,
            pipeline_stage_to_device_map=self.pipeline_stage_to_device_map,
        )

    def export_compile_run_layer(self, layer: CachedRotaryLayer, xq: AnyTensor) -> None:
        mlir_path = self._temp_dir / "rotary_layer.mlir"
        vmfb_path = self._temp_dir / "rotary_layer.vmfb"

        pipelined = isinstance(xq, ReplicatedTensor)
        devices = list(xq.devices) if pipelined else [0]
        assert len(devices) == 1, "Tensor parallelism not supported."

        arg_affinities = {0: aot.DeviceAffinity(devices[0])}  # xq

        fxb = aot.FxProgramsBuilder(layer)

        xq_or_shards = xq.shards if pipelined else xq

        @fxb.export_program(
            name=f"rotary_embedding",
            args=(xq_or_shards,),
            kwargs={},
            strict=False,
            arg_device=arg_affinities,
        )
        def _(module, xq_or_shards: list[torch.Tensor]):
            if pipelined:
                xq_or_shards = ReplicatedTensor(ts=xq_or_shards, devices=xq.devices)
            return module(xt=xq_or_shards)

        bundle = aot.export(fxb)
        bundle.save_mlir(mlir_path)

        extra_args = [
            f"-iree-hip-target={self.iree_hip_target}",
            "--iree-opt-level=O3",
            "--iree-dispatch-creation-propagate-collapse-across-expands=true",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hal-memoization=true",
            "--iree-codegen-enable-default-tuning-specs=true",
            "--iree-stream-affinity-solver-max-iterations=1024",
        ]
        extra_args.extend(
            f"--iree-hal-target-device=hip[{d}]" for d in range(max(devices) + 1)
        )

        compile_file(
            str(mlir_path),
            output_file=str(vmfb_path),
            extra_args=extra_args,
        )

        iree_devices = [
            iree.runtime.get_device(f"hip://{d}") for d in range(max(devices) + 1)
        ]
        instance = iree.runtime.VmInstance()
        hal = iree.runtime.create_hal_module(instance=instance, devices=iree_devices)

        vm_module = iree.runtime.VmModule.mmap(instance, str(vmfb_path.absolute()))
        modules = [hal, vm_module]
        context = iree.runtime.VmContext(instance=instance, modules=modules)

        forward = modules[-1].lookup_function("rotary_embedding")

        invoker = iree.runtime.FunctionInvoker(
            vm_context=context,
            device=iree_devices[0],
            vm_function=forward,
        )

        _xq = xq.shards[0] if pipelined else xq
        func_input = tensor_to_device_array(_xq, iree_devices[0])
        result_compiled = invoker(func_input)
        result_compiled = device_array_to_host(result_compiled).clone().detach()
        return result_compiled

    def test_rotary_table_eager_unsharded(self):
        rotary_layer = self.create_rotary_layer()
        em = rotary_layer(xt=self.xq)
        self.validate(em, self.xq)

    def test_rotary_table_eager_replicated(self):
        self.pipeline_stage_to_device_map = [[0], [1], [2], [3], [4], [5], [6], [7]]
        rotary_layer = self.create_rotary_layer()
        for devices in self.pipeline_stage_to_device_map:
            xq = ReplicatedTensor(ts=[self.xq], devices=devices)
            em = rotary_layer(xt=xq)
            assert em.devices == xq.devices
            self.validate(em, xq)

    @pytest.mark.skipif("config.getoption('iree_hal_target_device') != 'hip'")
    def test_rotary_table_iree_unsharded(self):
        rotary_layer = self.create_rotary_layer()

        result_compiled = self.export_compile_run_layer(rotary_layer, self.xq)
        self.validate(result_compiled, self.xq)

        result_eager = rotary_layer(xt=self.xq)
        self.validate(result_eager, self.xq)

        assert_tensor_close(result_eager, result_compiled, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif("config.getoption('iree_hal_target_device') != 'hip'")
    def test_rotary_table_iree_replicated(self):
        self.pipeline_stage_to_device_map = [[0], [1], [2], [3], [4], [5], [6], [7]]

        rotary_layer = self.create_rotary_layer()

        for devices in self.pipeline_stage_to_device_map:
            xq = ReplicatedTensor(ts=[self.xq], devices=devices)

            result_compiled = self.export_compile_run_layer(rotary_layer, xq)
            self.validate(result_compiled, xq)

            result_eager = rotary_layer(xt=xq)
            self.validate(result_eager, xq)

            assert_tensor_close(result_eager, result_compiled, atol=1e-4, rtol=1e-4)
