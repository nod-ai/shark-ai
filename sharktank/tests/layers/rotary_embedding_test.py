# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import math
import torch

import iree.runtime
import iree.turbine.aot as aot
from iree.compiler import compile_file

from parameterized import parameterized_class

from sharktank import ops
from sharktank.layers import CachedRotaryLayer, build_rotary_layer
from sharktank.types import AnyTensor, ReplicatedTensor
from sharktank.utils.iree import device_array_to_host, tensor_to_device_array
from sharktank.utils.testing import TempDirTestBase, is_mi300x


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
        em_01 = torch.transpose(em_01, -2, -1)
    else:
        xq_01 = xq.unflatten(-1, (2, rope_dims // 2))
        em_01 = em.unflatten(-1, (2, rope_dims // 2))
        xq_01 = torch.transpose(xq_01, -2, -1)
        em_01 = torch.transpose(em_01, -2, -1)

    xq_0 = xq_01[:, :, :, :, 0]
    xq_1 = xq_01[:, :, :, :, 1]

    em_0 = em_01[:, :, :, :, 0]
    em_1 = em_01[:, :, :, :, 1]

    xq_l = torch.sqrt(xq_0 * xq_0 + xq_1 * xq_1)
    em_l = torch.sqrt(em_0 * em_0 + em_1 * em_1)
    torch.testing.assert_close(xq_l, em_l)

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
    step = torch.where(step > math.pi * 2.0, step - math.pi * 2.0, step)
    step = torch.where(step < 0.0, step + math.pi * 2.0, step)

    # Check that the step size is approximately correct
    expected_step = torch.log(torch.asarray(rope_freq_base)) * (
        -(torch.arange(rope_dims // 2)) / (rope_dims // 2)
    )
    expected_step = torch.exp(expected_step)
    torch.testing.assert_close(step.flatten(), expected_step, atol=1e-2, rtol=1e-2)

    # Guarantee a progressive stepping for rotation:
    angle = angle / step
    angle = angle[:, 1:, ::]
    angle = torch.where(angle < 0, angle + math.pi * 2.0, angle)
    torch.testing.assert_close(
        angle, torch.full(angle.shape, 1.0), atol=1e-2, rtol=1e-2
    )


@parameterized_class(
    ("use_hf",),
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
            interleaved=(not self.use_hf),
        )

    def create_rotary_layer(self) -> CachedRotaryLayer:
        return build_rotary_layer(
            rope_dimension_count=self.rope_dims,
            rope_freq_base=self.rope_freq_base,
            use_hf=self.use_hf,
            pipeline_stage_to_device_map=self.pipeline_stage_to_device_map,
        )

    def export_compile_run_layer(self, layer: CachedRotaryLayer) -> None:
        mlir_path = self._temp_dir / "rotary_layer.mlir"
        vmfb_path = self._temp_dir / "rotary_layer.vmfb"

        fxb = aot.FxProgramsBuilder(layer)

        @fxb.export_program(
            name=f"forward_bs{self.bs}",
            args=(),
            kwargs={"xt": self.xq},
            strict=False,
        )
        def _(module, **kwargs):
            return module(**kwargs)

        bundle = aot.export(fxb)
        bundle.save_mlir(mlir_path)

        extra_args = [
            "-iree-hip-target=gfx942",
            "--iree-opt-level=O3",
            "--iree-dispatch-creation-propagate-collapse-across-expands=true",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hal-memoization=true",
            "--iree-codegen-enable-default-tuning-specs=true",
            "--iree-stream-affinity-solver-max-iterations=1024",
        ]
        devices = (
            sorted(set(itertools.chain(*self.pipeline_stage_to_device_map)))
            if self.pipeline_stage_to_device_map
            else [0]
        )
        extra_args.extend(f"--iree-hal-target-device=hip[{d}]" for d in devices)

        compile_file(
            str(mlir_path),
            output_file=str(vmfb_path),
            extra_args=extra_args,
        )

        iree_devices = [iree.runtime.get_device(f"hip://{d}") for d in devices]
        instance = iree.runtime.VmInstance()
        hal = iree.runtime.create_hal_module(instance=instance, devices=iree_devices)

        vm_module = iree.runtime.VmModule.mmap(instance, str(vmfb_path.absolute()))
        modules = [hal, vm_module]
        context = iree.runtime.VmContext(instance=instance, modules=modules)

        forward = modules[-1].lookup_function("forward_bs1")

        invoker = iree.runtime.FunctionInvoker(
            vm_context=context,
            device=iree_devices[0],
            vm_function=forward,
        )

        func_input = tensor_to_device_array(self.xq, iree_devices[0])
        result_compiled = invoker(func_input)
        result_compiled = device_array_to_host(result_compiled).clone().detach()
        return result_compiled

    def test_rotary_table_eager_unsharded(self):
        rotary_layer = self.create_rotary_layer()
        em = rotary_layer(xt=self.xq)
        self.validate(em, self.xq)

    def test_rotary_table_eager_replicated(self):
        self.pipeline_stage_to_device_map = [[0], [0], [1], [1]]

        rotary_layer = self.create_rotary_layer()
        for devices in self.pipeline_stage_to_device_map:
            xq = ReplicatedTensor(ts=[self.xq], devices=devices)
            em = rotary_layer(xt=xq)
            assert em.devices == xq.devices
            self.validate(em, xq)

    # TODO: Add parallelism
    @is_mi300x
    def test_rotary_export_compile_and_run(self):
        """Export layer to MLIR, compile with IREE, run compiled module, and validate output."""
        layer = self.create_rotary_layer()

        result_compiled = self.export_compile_run_layer(layer)
        self.validate(result_compiled, self.xq)

        result_eager = layer(xt=self.xq)
        self.validate(result_eager, self.xq)

        torch.testing.assert_close(
            ops.unshard(result_eager),
            ops.unshard(result_compiled),
            atol=1e-4,
            rtol=1e-4,
        )
