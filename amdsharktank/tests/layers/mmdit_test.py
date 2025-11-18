# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

import unittest

import torch

from iree.turbine import aot
from amdsharktank.layers import (
    MMDITDoubleBlock,
    MMDITSingleBlock,
)
from amdsharktank.layers.testing import (
    make_mmdit_double_block_random_theta,
    make_mmdit_single_block_random_theta,
)
from amdsharktank.utils.testing import TempDirTestBase
from amdsharktank.types import Dataset, Theta


class MMDITTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)
        self.hidden_size = 3072
        self.num_heads = 24
        self.batch_size = 3

    def testDoubleExport(self):
        theta = make_mmdit_double_block_random_theta(hidden_size=self.hidden_size)
        theta = self.save_load_theta(theta)
        mmdit = MMDITDoubleBlock(
            theta=theta, num_heads=self.num_heads, hidden_size=self.hidden_size
        )

        img = torch.rand([self.batch_size, 1024, self.hidden_size])
        txt = torch.rand([self.batch_size, 512, self.hidden_size])
        vec = torch.rand([self.batch_size, self.hidden_size])
        rot = torch.rand([self.batch_size, 1, 1536, 64, 2, 2])
        mmdit.forward(img, txt, vec, rot)
        fxb = aot.FxProgramsBuilder(mmdit)

        @fxb.export_program(name="mmdit", args=(img, txt, vec, rot), strict=False)
        def _(model, img, txt, vec, rot) -> torch.Tensor:
            return model.forward(img, txt, vec, rot)

        output = aot.export(fxb)
        output.verify()
        asm = str(output.mlir_module)

    def testSingleExport(self):
        mlp_ratio = 4.0
        theta = make_mmdit_single_block_random_theta(
            hidden_size=self.hidden_size, num_heads=self.num_heads, mlp_ratio=mlp_ratio
        )
        theta = self.save_load_theta(theta)
        mmdit = MMDITSingleBlock(
            theta=theta,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            mlp_ratio=mlp_ratio,
        )

        inp = torch.rand([self.batch_size, 1024, self.hidden_size])
        vec = torch.rand([self.batch_size, self.hidden_size])
        rot = torch.rand([self.batch_size, 1, 1024, 64, 2, 2])
        mmdit.forward(inp, vec, rot)
        fxb = aot.FxProgramsBuilder(mmdit)

        @fxb.export_program(name="mmdit", args=(inp, vec, rot), strict=False)
        def _(model, inp, vec, rot) -> torch.Tensor:
            return model.forward(inp, vec, rot)

        output = aot.export(fxb, import_symbolic_shape_expressions=True)
        output.verify()
        asm = str(output.mlir_module)

    def save_load_theta(self, theta: Theta):
        # Roundtrip to disk to avoid treating parameters as constants that would appear
        # in the MLIR.
        theta.rename_tensors_to_paths()
        dataset = Dataset(root_theta=theta, properties={})
        file_path = self._temp_dir / "parameters.irpa"
        dataset.save(file_path)
        return Dataset.load(file_path).root_theta


if __name__ == "__main__":
    unittest.main()
