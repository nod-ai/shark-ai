# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from datetime import datetime
import os
import sys
import unittest
import pytest
import subprocess
from pathlib import Path
from typing import List
from sharktank.utils.export_artifacts import (
    ExportArtifacts,
    ExportMlirException,
    IreeBenchmarkException,
    IreeCompileException,
)

is_mi300x = pytest.mark.skipif("config.getoption('iree_hip_target') != 'gfx942'")
skipif_run_quick_llama_test = pytest.mark.skipif(
    'config.getoption("run-quick-llama-test") and not config.getoption("run-nightly-llama-tests")',
    reason="Skipping largs tests when --run-quick-llama-test is set.",
)


@pytest.mark.usefixtures("get_iree_flags")
class BaseBenchmarkTest(unittest.TestCase):
    directory_created = False
    current_date = datetime.now()
    dir_path_suffix = current_date.strftime("%Y-%m-%d")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.dirname(cur_dir)
    tests_dir = os.path.dirname(models_dir)
    sharktank_dir = os.path.dirname(tests_dir)
    repo_root = os.path.dirname(sharktank_dir)
    dir_path = Path(repo_root + "/" + dir_path_suffix)

    @classmethod
    def setUpClass(cls):
        """This method will be run once per class to create the directory."""
        if not cls.directory_created:
            if not os.path.exists(cls.dir_path):
                os.makedirs(cls.dir_path)
            cls.directory_created = True

    def setUp(self):
        self.compile_args = [
            "--iree-dispatch-creation-enable-aggressive-fusion=true",
            "--iree-global-opt-propagate-transposes=true",
            "--iree-opt-aggressively-propagate-transposes=true",
            "--iree-opt-data-tiling=false",
            "--iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))'",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-hal-memoization=true",
            "--iree-opt-strip-assertions",
        ]


@is_mi300x
class BenchmarkLlama3_1_8B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/8b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path_tp1 = self.weights_dir / "llama3.1_8b_instruct_fp16.irpa"
        self.irpa_path_tp8 = self.weights_dir / "tp8/llama3.1_8b_instruct_fp16_tp8.irpa"
        self.irpa_path_fp8 = (
            self.artifacts_dir / "fp8/native_fp8_e4m3fnuz_llama3_8b.irpa"
        )
        self.dir_path_8b = self.dir_path / "llama-8b"
        self.temp_dir_8b = Path(self.dir_path_8b)
        self.temp_dir_8b.mkdir(parents=True, exist_ok=True)
        self.llama8b_f16_torch_sdpa_artifacts_tp1 = ExportArtifacts(
            irpa_path=str(self.irpa_path_tp1),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            block_seq_stride=32,
        )
        self.llama8b_fp8_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            block_seq_stride=32,
            activation_dtype="bfloat16",
            attention_dtype="float8_e4m3fnuz",
        )
        self.prefill_args_bs4_128_f16_tp1 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp1"
        )
        self.decode_args_bs4_128_f16_tp1 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32_tp1"
        )
        self.prefill_args_bs4_2048_f16_tp1 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32"
        )
        self.decode_args_bs4_2048_f16_tp1 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32"
        )
        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
        # TODO: make function for prefill and decode args
        self.iree_run_prefill_args_fp16_128_tp1 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/tokens.npy",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/seq_lens.npy",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_f16_128_tp1 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/next_tokens.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/seq_lens.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/start_positions.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_prefill_args_f16_2048_tp1 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/tokens.npy",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/seq_lens.npy",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_f16_2048_tp1 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/next_tokens.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/seq_lens.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/start_positions.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_fp8}/tokens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_lens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_fp8}/tokens.npy",
            f"--input=@{self.decode_args_fp8}/seq_lens.npy",
            f"--input=@{self.decode_args_fp8}/start_positions.npy",
            f"--input=@{self.decode_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.decode_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    def testBenchmark8B_f16_TP1_Input_Len_128(self):
        output_file_name = self.dir_path_8b / "f16_torch_128_tp1"
        output_mlir = self.llama8b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        export_return_code = self.llama8b_f16_torch_sdpa_artifacts_tp1.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_f16_torch_sdpa_artifacts_tp1.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama8b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_prefill_args_fp16_128_tp1,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_decode_args_f16_128_tp1,
            cwd=self.repo_root,
        )

    @skipif_run_quick_llama_test
    def testBenchmark8B_f16_TP1_Input_Len_2048(self):
        output_file_name = self.dir_path_8b / "f16_torch_2048_tp1"
        output_mlir = self.llama8b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        export_return_code = self.llama8b_f16_torch_sdpa_artifacts_tp1.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_f16_torch_sdpa_artifacts_tp1.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama8b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_prefill_args_f16_2048_tp1,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_decode_args_f16_2048_tp1,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="Benchmark inputs not configured yet.",
        strict=True,
        raises=IreeBenchmarkException,
    )
    def testBenchmark8B_fp8_Non_Decomposed(self):
        output_file_name = self.dir_path_8b / "fp8_torch"
        output_mlir = self.llama8b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        export_return_code = self.llama8b_fp8_torch_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_fp8_torch_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama8b_fp8_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args_fp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_fp8_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args_fp8,
            cwd=self.repo_root,
        )


@is_mi300x
@skipif_run_quick_llama_test
class BenchmarkLlama3_1_70B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/70b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path_tp1 = self.weights_dir / "llama3.1_70b_instruct_fp16.irpa"
        self.irpa_path_tp8 = self.weights_dir / "tp8/llama3_70b_instruct_fp16_tp8.irpa"
        self.irpa_path_fp8 = self.artifacts_dir / "f8/llama70b_fp8.irpa"
        self.dir_path_70b = self.dir_path / "llama-70b"
        self.temp_dir_70b = Path(self.dir_path_70b)
        self.temp_dir_70b.mkdir(parents=True, exist_ok=True)
        self.llama70b_f16_torch_sdpa_artifacts_tp1 = ExportArtifacts(
            irpa_path=str(self.irpa_path_tp1),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            block_seq_stride=32,
        )
        self.llama70b_f16_torch_sdpa_artifacts_tp8 = ExportArtifacts(
            irpa_path=str(self.irpa_path_tp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=8,
            block_seq_stride=32,
        )
        self.llama70b_fp8_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            block_seq_stride=32,
        )
        self.prefill_args_bs4_128_f16_tp1 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32"
        )
        self.decode_args_bs4_128_f16_tp1 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32"
        )
        self.prefill_args_bs4_2048_f16_tp1 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32"
        )
        self.decode_args_bs4_2048_f16_tp1 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32"
        )
        self.prefill_args_bs4_128_f16_tp8 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp8"
        )
        self.decode_args_bs4_128_f16_tp8 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32_tp8"
        )
        self.prefill_args_bs4_2048_f16_tp8 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32_tp8"
        )
        self.decode_args_bs4_2048_f16_tp8 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32_tp8"
        )
        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_args_128_f16_tp1 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/tokens.npy",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/seq_lens.npy",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_bs4_128_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_128_f16_tp1 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/next_tokens.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/seq_lens.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/start_positions.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.decode_args_bs4_128_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_prefill_args_2048_f16_tp1 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/tokens.npy",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/seq_lens.npy",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_bs4_2048_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_2048_f16_tp1 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/next_tokens.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/seq_lens.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/start_positions.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/seq_block_ids.npy",
            f"--input=@{self.decode_args_bs4_2048_f16_tp1}/cs_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_prefill_args_128_f16_tp8 = (
            [
                "--function=prefill_bs4",
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/tokens.npy",
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/seq_lens.npy",
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(8)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_decode_args_128_f16_tp8 = (
            [
                "--function=decode_bs4",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/next_tokens.npy",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/seq_lens.npy",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/start_positions.npy",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(8)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_prefill_args_2048_f16_tp8 = (
            [
                "--function=prefill_bs4",
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/tokens.npy",
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/seq_lens.npy",
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(8)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_decode_args_2048_f16_tp8 = (
            [
                "--function=decode_bs4",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/next_tokens.npy",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/seq_lens.npy",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/start_positions.npy",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(8)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_fp8}/tokens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_lens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_fp8}/tokens.npy",
            f"--input=@{self.decode_args_fp8}/seq_lens.npy",
            f"--input=@{self.decode_args_fp8}/start_positions.npy",
            f"--input=@{self.decode_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.decode_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    def testBenchmark70B_f16_TP1_Input_Len_128(self):
        output_file_name = self.dir_path_70b / "f16_torch_128_tp1"
        output_mlir = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        export_return_code = self.llama70b_f16_torch_sdpa_artifacts_tp1.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama70b_f16_torch_sdpa_artifacts_tp1.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama70b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_prefill_args_128_f16_tp1,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_decode_args_128_f16_tp1,
            cwd=self.repo_root,
        )

    def testBenchmark70B_f16_TP1_Input_Len_2048(self):
        output_file_name = self.dir_path_70b / "f16_torch_2048_tp1"
        output_mlir = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        export_return_code = self.llama70b_f16_torch_sdpa_artifacts_tp1.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama70b_f16_torch_sdpa_artifacts_tp1.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama70b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_prefill_args_2048_f16_tp1,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp1,
            args=self.iree_run_decode_args_2048_f16_tp1,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark70B_f16_TP8_Input_Len_128(self):
        output_file_name = self.dir_path_70b / "f16_torch_128_tp8"
        output_mlir = self.llama70b_f16_torch_sdpa_artifacts_tp8.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama70b_f16_torch_sdpa_artifacts_tp8.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama70b_f16_torch_sdpa_artifacts_tp8.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_shard_file_name = (
            self.weights_dir
            / f"tp8/llama3_70b_instruct_fp16_tp{self.llama70b_f16_torch_sdpa_artifacts_tp8.tensor_parallelism_size}.irpa"
        )
        if output_shard_file_name.exists():
            self.llama70b_f16_torch_sdpa_artifacts_tp8.irpa_path = (
                output_shard_file_name
            )
        export_return_code = self.llama70b_f16_torch_sdpa_artifacts_tp8.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama70b_f16_torch_sdpa_artifacts_tp8.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama70b_f16_torch_sdpa_artifacts_tp8.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_prefill_args_128_f16_tp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp8.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_decode_args_128_f16_tp8,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark70B_f16_TP8_Input_Len_2048(self):
        output_file_name = self.dir_path_70b / "f16_torch_2048_tp8"
        output_mlir = self.llama70b_f16_torch_sdpa_artifacts_tp8.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama70b_f16_torch_sdpa_artifacts_tp8.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama70b_f16_torch_sdpa_artifacts_tp8.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_shard_file_name = (
            self.weights_dir
            / f"tp8/llama3_70b_instruct_fp16_tp{self.llama70b_f16_torch_sdpa_artifacts_tp8.tensor_parallelism_size}.irpa"
        )
        if output_shard_file_name.exists():
            self.llama70b_f16_torch_sdpa_artifacts_tp8.irpa_path = (
                output_shard_file_name
            )
        export_return_code = self.llama70b_f16_torch_sdpa_artifacts_tp8.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama70b_f16_torch_sdpa_artifacts_tp8.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama70b_f16_torch_sdpa_artifacts_tp8.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_prefill_args_2048_f16_tp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp8.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_decode_args_2048_f16_tp8,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="70b fp8 irpa does not exist", strict=True, raises=ExportMlirException
    )
    def testBenchmark70B_fp8_TP8(self):
        output_file_name = self.dir_path_70b / "fp8_torch"
        output_mlir = self.llama70b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama70b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama70b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_shard_file_name = (
            self.artifacts_dir
            / f"f8/tp8/llama3.1_70b_fp8_tp{self.llama70b_fp8_torch_sdpa_artifacts.tensor_parallelism_size}_parameters.irpa"
        )
        if output_shard_file_name.exists():
            self.llama70b_fp8_torch_sdpa_artifacts.irpa_path = output_shard_file_name
        export_return_code = self.llama70b_fp8_torch_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama70b_fp8_torch_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama70b_fp8_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_fp8_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )


@is_mi300x
@skipif_run_quick_llama_test
class BenchmarkLlama3_1_405B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/405b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path_tp1 = self.weights_dir / "llama3_405b_instruct_fp16.irpa"
        self.irpa_path_tp8 = self.weights_dir / "tp8/llama3_405b_instruct_fp16_tp8.irpa"
        self.irpa_path_fp8 = self.artifacts_dir / "f8/llama3.1_405b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.dir_path_405b = self.dir_path / "llama-405b"
        self.temp_dir_405b = Path(self.dir_path_405b)
        self.temp_dir_405b.mkdir(parents=True, exist_ok=True)
        self.llama405b_f16_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_tp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            block_seq_stride=32,
        )
        self.llama405b_fp8_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            block_seq_stride=32,
        )
        self.prefill_args_bs4_128_f16_tp8 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp8"
        )
        self.decode_args_bs4_128_f16_tp8 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32_tp8"
        )
        self.prefill_args_bs4_2048_f16_tp8 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32_tp8"
        )
        self.decode_args_bs4_2048_f16_tp8 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32_tp8"
        )
        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_args_128_f16_tp8 = (
            [
                "--function=prefill_bs4",
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/tokens.npy",
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/seq_lens.npy",
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.prefill_args_bs4_128_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(self.tensor_parallelism_size)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_decode_args_128_f16_tp8 = (
            [
                "--function=decode_bs4",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/next_tokens.npy",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/seq_lens.npy",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/start_positions.npy",
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.decode_args_bs4_128_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(self.tensor_parallelism_size)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_prefill_args_2048_f16_tp8 = (
            [
                "--function=prefill_bs4",
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/tokens.npy",
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/seq_lens.npy",
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.prefill_args_bs4_2048_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(self.tensor_parallelism_size)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_decode_args_2048_f16_tp8 = (
            [
                "--function=decode_bs4",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/next_tokens.npy",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/seq_lens.npy",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/start_positions.npy",
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/seq_block_ids.npy",
            ]
            + [
                f"--input=@{self.decode_args_bs4_2048_f16_tp8}/cs_f16_shard_{i}.npy"
                for i in range(self.tensor_parallelism_size)
            ]
            + [
                "--benchmark_repetitions=3",
            ]
        )
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{self.prefill_args_fp8}/tokens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_lens.npy",
            f"--input=@{self.prefill_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.prefill_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{self.decode_args_fp8}/tokens.npy",
            f"--input=@{self.decode_args_fp8}/seq_lens.npy",
            f"--input=@{self.decode_args_fp8}/start_positions.npy",
            f"--input=@{self.decode_args_fp8}/seq_block_ids.npy",
            f"--input=@{self.decode_args_fp8}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark405B_f16_TP8_Input_Len_128(self):
        output_file_name = self.dir_path_405b / "f16_torch_128_tp8"
        output_mlir = self.llama405b_f16_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama405b_f16_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama405b_f16_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_shard_file_name = (
            self.weights_dir
            / f"tp8/llama3_405b_instruct_fp16_tp{self.tensor_parallelism_size}.irpa"
        )
        if output_shard_file_name.exists():
            self.llama405b_f16_torch_sdpa_artifacts.irpa_path = output_shard_file_name
        export_return_code = self.llama405b_f16_torch_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama405b_f16_torch_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama405b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_prefill_args_128_f16_tp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama405b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_decode_args_128_f16_tp8,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark405B_f16_TP8_Input_Len_2048(self):
        output_file_name = self.dir_path_405b / "f16_torch_2048_tp8"
        output_mlir = self.llama405b_f16_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama405b_f16_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama405b_f16_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_shard_file_name = (
            self.weights_dir
            / f"tp8/llama3_405b_instruct_fp16_tp{self.tensor_parallelism_size}.irpa"
        )
        if output_shard_file_name.exists():
            self.llama405b_f16_torch_sdpa_artifacts.irpa_path = output_shard_file_name
        export_return_code = self.llama405b_f16_torch_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama405b_f16_torch_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama405b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_prefill_args_2048_f16_tp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama405b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_tp8,
            args=self.iree_run_decode_args_2048_f16_tp8,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="KeyError in theta.py", strict=True, raises=ExportMlirException
    )
    def testBenchmark405B_fp8_TP8(self):
        output_file_name = self.dir_path_405b / "fp8_torch"
        output_mlir = self.llama405b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama405b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama405b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_shard_file_name = (
            self.artifacts_dir
            / f"f8/tp8/llama3.1_405b_fp8_tp{self.tensor_parallelism_size}_parameters.irpa"
        )
        if output_shard_file_name.exists():
            self.llama405b_fp8_torch_sdpa_artifacts.irpa_path = output_shard_file_name
        export_return_code = self.llama405b_fp8_torch_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama405b_fp8_torch_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama405b_fp8_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama405b_fp8_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )


if __name__ == "__main__":
    unittest.main()
