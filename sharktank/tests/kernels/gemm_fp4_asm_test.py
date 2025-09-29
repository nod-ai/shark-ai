# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
import iree.compiler as ireec
import iree.runtime as ireert
import iree.turbine.aot as aot
import numpy as np
from pathlib import Path
from sharktank.kernels.gemm_fp4_asm import asm_fp4_gemm, shuffle_weight
from sharktank.types.quantizers import DynamicFp4BlockQuantizer
from sharktank.utils.testing import assert_cosine_similarity_close, is_mi350x, IreeFlags

logging.basicConfig(level=logging.DEBUG)


@is_mi350x
@pytest.mark.usefixtures("iree_flags")
class TestAsmFp4Gemm:
    def hip_flags(self):
        return [
            "--iree-hip-target=gfx950",
            "--iree-hal-target-device=hip",
            "--iree-hal-target-backends=rocm",
            "--iree-hip-specialize-dispatches",
            "--iree-opt-level=O3",
            "--iree-codegen-enable-default-tuning-specs=true",
            "--iree-dispatch-creation-enable-early-trunc-fusion=true",
            "--iree-dispatch-creation-propagate-collapse-across-expands=true",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-hal-memoization=true",
            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            "--iree-vm-target-index-bits=64",
            "--iree-stream-affinity-solver-max-iterations=1024",
            "--iree-stream-resource-index-bits=64",
            "--iree-stream-resource-max-allocation-size=4294967296",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hip-encoding-layout-resolver=data-tiling",
        ]

    @is_mi350x
    @pytest.mark.parametrize(
        "m, n, k, use_preshuffle",
        [
            (256, 256, 1024, False),
            (256, 256, 1024, True),
            (256, 2048, 8192, False),
            (256, 2048, 8192, True),
        ],
    )
    def test_asm_fp4_gemm_export_compile_run(
        self,
        deterministic_random_seed,
        iree_flags: IreeFlags,
        tmp_path: Path,
        m: int,
        n: int,
        k: int,
        use_preshuffle: bool,
    ):
        assert k % 32 == 0

        class AsmMxfp4GemmModule(torch.nn.Module):
            def forward(self, x, w, x_scale, w_scale, bias):
                return asm_fp4_gemm(
                    x, w, x_scale, w_scale, bias, use_preshuffle=use_preshuffle
                )

        e = aot.export(
            AsmMxfp4GemmModule(),
            args=(
                torch.empty((m, k // 2), dtype=torch.uint8),
                torch.empty((n, k // 2), dtype=torch.uint8),
                torch.empty((m, k // 32), dtype=torch.uint8),
                torch.empty((n, k // 32), dtype=torch.uint8),
                torch.empty((m, n), dtype=torch.float32),
            ),
        )
        e.verify()
        mlir_asm = str(e.mlir_module)
        assert "func.func @main" in mlir_asm
        assert "util.func private @asm_mxfp4_gemm" in mlir_asm
        assert "util.func private @shuffle_scales" in mlir_asm

        mlir_path = tmp_path / "asm_fp4_gemm.mlir"
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

        lhs = torch.randn(m, k, dtype=torch.float32)
        rhs = torch.randn(k, n, dtype=torch.float32)
        expected = lhs @ rhs

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
        bias = torch.zeros(m, n, dtype=torch.float32)

        if use_preshuffle:
            w = shuffle_weight(w_t, layout=(16, 16))
        else:
            w = w_t

        _asm_fp4_gemm_main = modules[-1].main
        iree_results = _asm_fp4_gemm_main(x, w, x_scales, w_scales, bias)
        iree_results = torch.from_numpy(
            np.asarray(iree_results.to_host()).astype(np.float16)
        ).to(torch.float32)
        assert_cosine_similarity_close(iree_results, expected, atol=0.05)
