# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import json
import pandas
import pytest
import sys
import torch

from sharktank.models.flux.benchmark import iree_benchmark_flux_dev_transformer
from sharktank.models.flux.testing import with_flux_data
from sharktank.utils.benchmark import (
    iree_benchmark_compare,
    iree_benchmark_assert_contender_is_not_worse,
)
from sharktank.utils.testing import is_mi300x
from sharktank.utils.testing import TempDirTestBase


@pytest.mark.usefixtures("get_iree_flags", "caching", "path_prefix")
@pytest.mark.benchmark
@pytest.mark.expensive
class FluxBenchmark(TempDirTestBase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(12345)
        if self.path_prefix is None:
            self.path_prefix = self._temp_dir

    @is_mi300x
    @with_flux_data
    def testBenchmarkFluxDevTransformerMi300x(self):
        benchmark_result_file_path = Path(self.path_prefix) / "benchmark_result.json"
        benchmark_result = iree_benchmark_flux_dev_transformer(
            artifacts_dir=Path(self.path_prefix),
            iree_device=self.iree_device,
            json_result_output_path=benchmark_result_file_path,
            caching=self.caching,
        )
        print(benchmark_result)

        baseline_benchmark_result_file_path = (
            Path(__file__).parent
            / "flux_transformer_baseline_benchmark_result_mi300x.json"
        )
        benchmark_compare_result_file_path = (
            Path(self.path_prefix) / "benchmark_compare_result.json"
        )

        benchmark_compare_result = iree_benchmark_compare(
            [
                f"--dump_to_json={benchmark_compare_result_file_path}",
                "benchmarks",
                str(baseline_benchmark_result_file_path),
                benchmark_result_file_path,
            ]
        )
        print(benchmark_compare_result)

        with open(benchmark_compare_result_file_path, "r") as f:
            benchmark_compare_result = json.load(f)
        iree_benchmark_assert_contender_is_not_worse(benchmark_compare_result)
