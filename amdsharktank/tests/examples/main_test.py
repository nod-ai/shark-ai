# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys
import unittest

from amdsharktank.utils.testing import MainRunnerTestBase


@pytest.mark.skipif(
    sys.platform == "win32", reason="https://github.com/nod-ai/amdshark-ai/issues/698"
)
class ShardingTests(MainRunnerTestBase):
    def testExportFfnNet(self):
        from amdsharktank.examples.sharding.export_ffn_net import main

        irpa_path = self.get_irpa_path("ffn")
        output_path = self.get_file_path("output.mlir")

        self.run_main(main, "--output-irpa-file", irpa_path, output_path)
        self.assertFileWritten(irpa_path)
        self.assertFileWritten(output_path)


@pytest.mark.skipif(
    sys.platform == "win32", reason="https://github.com/nod-ai/amdshark-ai/issues/698"
)
class PipelineTests(MainRunnerTestBase):
    def testExportPFfnNet(self):
        from amdsharktank.examples.pipeline.export_ppffn_net import main

        irpa_path = self.get_irpa_path("ppffn")
        output_path = self.get_file_path("output_ppffn.mlir")

        self.run_main(main, "--output-irpa-file", irpa_path, output_path)
        self.assertFileWritten(irpa_path)
        self.assertFileWritten(output_path)


if __name__ == "__main__":
    unittest.main()
