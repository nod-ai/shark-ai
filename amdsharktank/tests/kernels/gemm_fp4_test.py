# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import unittest

import iree.compiler as ireec
import iree.runtime as ireert
import iree.turbine.aot as aot
import numpy as np
from pathlib import Path
from amdsharktank.kernels.gemm_fp4 import *
from amdsharktank.types.layout_utils import (
    pack_fp4_e2m1_to_uint8,
    unpack_uint8_to_fp4_e2m1,
)
from amdsharktank.types.ocp_floats import (
    e8m0_to_float32,
    fp4_e2m1_to_float32,
)
from amdsharktank.types.quantizers import DynamicFp4BlockQuantizer
from amdsharktank.utils.testing import (
    assert_cosine_similarity_close,
    is_mi350x,
    IreeFlags,
)


def _reference_batched_block_scaled_mmt_fp4(
    x_packed: torch.Tensor,
    x_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference implementation for batched block scaled FP4 MMT."""

    x_unpacked = unpack_uint8_to_fp4_e2m1(x_packed)
    w_unpacked = unpack_uint8_to_fp4_e2m1(w_packed)

    x_f32 = fp4_e2m1_to_float32(x_unpacked)
    w_f32 = fp4_e2m1_to_float32(w_unpacked)
    x_scales_f32 = e8m0_to_float32(x_scales)
    w_scales_f32 = e8m0_to_float32(w_scales)

    x_f32_scaled = x_f32 * x_scales_f32
    w_f32_scaled = w_f32 * w_scales_f32

    b, m, k = x_f32_scaled.shape
    n, _ = w_f32_scaled.shape
    x_flatten = torch.flatten(x_f32_scaled, start_dim=0, end_dim=1)
    res = torch.mm(x_flatten, w_f32_scaled.t())
    res = res.reshape(b, m, n)
    return res.to(out_dtype)


class MyModule(torch.nn.Module):
    def forward(self, x, x_scales, w, w_scales):
        return iree_mxfp4_bmm(x, x_scales, w, w_scales)


class TestMxfp4Kernel:
    def setUp(self):
        torch.manual_seed(42)

    def test_template(self):
        x = torch.randn([4, 8, 16]).to(torch.uint8)
        x_scales = torch.randn([4, 8, 1]).to(torch.uint8)
        w = torch.randn([4, 16]).to(torch.uint8)
        w_scales = torch.randn([4, 1]).to(torch.uint8)
        mod = MyModule()
        dtype = torch.dtype
        ep = torch.export.export(
            mod,
            args=(x, x_scales, w, w_scales),
        )
        output = aot.export(ep)
        mlir_src = str(output.mlir_module)
        assert "linalg.generic" in mlir_src
        assert "arith.scaling_extf" in mlir_src
        output.verify()

    @pytest.mark.parametrize(
        "b,m,n,k",
        [
            (4, 8, 4, 32),
            (4, 256, 256, 256),
        ],
    )
    def test_kernel_golden(
        self,
        b: int,
        m: int,
        n: int,
        k: int,
    ):
        x = torch.full([b, m, k // 2], 34).to(torch.uint8)
        x_scales = torch.full([b, m, k // 32], 127).to(torch.uint8)
        w = torch.full([n, k // 2], 34).to(torch.uint8)
        w_scales = torch.full([n, k // 32], 127).to(torch.uint8)
        res = iree_mxfp4_bmm(x, x_scales, w, w_scales)
        ref = torch.full([b, m, n], k).to(torch.float16)
        torch.testing.assert_close(res, ref, atol=5e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "b,m,n,k",
        [
            (4, 8, 4, 32),
        ],
    )
    def test_random(
        self,
        b: int,
        m: int,
        n: int,
        k: int,
    ):
        x = torch.randint(0, 16, (b, m, k)).to(torch.uint8)
        x_packed = pack_fp4_e2m1_to_uint8(x)
        w = torch.randint(0, 16, (n, k)).to(torch.uint8)
        w_packed = pack_fp4_e2m1_to_uint8(w)
        x_scales = torch.randint(125, 128, (b, m, k // 32)).to(torch.uint8)
        w_scales = torch.randint(125, 128, (n, k // 32)).to(torch.uint8)
        res = iree_mxfp4_bmm(x_packed, x_scales, w_packed, w_scales)
        ref = _reference_batched_block_scaled_mmt_fp4(
            x_packed, x_scales, w_packed, w_scales
        )
        torch.testing.assert_close(res, ref, atol=5e-3, rtol=1e-3)


@is_mi350x
@pytest.mark.usefixtures("iree_flags")
class TestKernelOnGpu:
    def setUp(self):
        torch.manual_seed(42)

    def hip_flags(self):
        return [
            "--iree-hip-target=gfx950",
            "--iree-hal-target-device=hip",
            "--iree-hal-target-backends=rocm",
            "--iree-opt-level=O3",
        ]

    @pytest.mark.parametrize(
        "b,m,n,k",
        [
            (4, 8, 4, 64),
            (4, 256, 256, 1024),
        ],
    )
    def test_quantization(
        self,
        iree_flags: IreeFlags,
        tmp_path: Path,
        b: int,
        m: int,
        n: int,
        k: int,
    ):
        B = torch.export.Dim("B")
        M = torch.export.Dim("M")
        e = aot.export(
            MyModule(),
            args=(
                torch.empty((b, m, k // 2), dtype=torch.uint8),
                torch.empty((b, m, k // 32), dtype=torch.uint8),
                torch.empty((n, k // 2), dtype=torch.uint8),
                torch.empty((n, k // 32), dtype=torch.uint8),
            ),
            dynamic_shapes={
                "x": {0: B, 1: M},
                "x_scales": {0: B, 1: M},
                "w": {},
                "w_scales": {},
            },
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        assert "func.func @main" in mlir_asm

        mlir_path = tmp_path / "fp4_gemm.mlir"
        with open(str(mlir_path), "w") as f:
            f.write(mlir_asm)
        vmfb = ireec.compile_file(
            str(mlir_path),
            extra_args=self.hip_flags(),
        )

        instance = ireert.VmInstance()
        devices = [ireert.get_device(iree_flags.iree_device)]
        config = ireert.Config(device=devices[0])
        hal = ireert.create_hal_module(instance, devices=devices)
        binary = ireert.VmModule.copy_buffer(instance, vmfb)
        modules = ireert.load_vm_modules(hal, binary, config=config)

        lhs = torch.randn((b, m, k), dtype=torch.float32)
        rhs = torch.randn((k, n), dtype=torch.float32)
        ref = lhs @ rhs

        quantizer = DynamicFp4BlockQuantizer(
            block_size=32, use_fe8m0_scale=True, name="matmul_input_quantizer"
        )
        lhs_quantized = quantizer.quantize(lhs)
        lhs_unpacked = lhs_quantized.unpack()
        rhs_quantized = quantizer.quantize(rhs.mT)
        rhs_unpacked = rhs_quantized.unpack()

        x = lhs_unpacked.qs_bit_packed.flatten(start_dim=-2)
        x_scales = lhs_unpacked.d.squeeze(-1)
        w_t = rhs_unpacked.qs_bit_packed.flatten(start_dim=-2)
        w_scales = rhs_unpacked.d.squeeze(-1)
        x_np = x.detach().cpu().numpy()
        x_scales_np = x_scales.detach().cpu().numpy()
        w_np = w_t.detach().cpu().numpy()
        w_scales_np = w_scales.detach().cpu().numpy()
        _fp4_gemm_main = modules[-1].main
        res = _fp4_gemm_main(x, x_scales, w_t, w_scales)
        res = torch.from_numpy(np.asarray(res.to_host()).astype(np.float16)).to(
            torch.float32
        )
        assert_cosine_similarity_close(res, ref, atol=0.05)
