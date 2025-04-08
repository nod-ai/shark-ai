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
            "--iree-opt-level=O3",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hal-memoization=true",
        ]

    def save_benchmarks(
        self,
        *,
        benchmark_fn: str,
        input_path: Path,
        tensor_parallelism_size=1,
        benchmark_repetitions=3,
    ):
        benchmark_args = [
            f"--function={benchmark_fn}",
        ]

        if "prefill" in benchmark_fn:
            benchmark_args += [
                f"--input=@{input_path}/tokens.npy",
                f"--input=@{input_path}/seq_lens.npy",
            ]
        elif "decode" in benchmark_fn:
            benchmark_args += [
                f"--input=@{input_path}/next_tokens.npy",
                f"--input=@{input_path}/seq_lens.npy",
                f"--input=@{input_path}/start_positions.npy",
            ]

        benchmark_args += [f"--input=@{input_path}/seq_block_ids.npy"]

        if tensor_parallelism_size == 1:
            benchmark_args += [
                f"--input=@{input_path}/cs_f16.npy",
            ]
        else:
            benchmark_args += [
                f"--input=@{input_path}/cs_f16_shard_{i}.npy"
                for i in range(tensor_parallelism_size)
            ]

        benchmark_args += [
            f"--benchmark_repetitions={benchmark_repetitions}",
            ">>",
        ]

        return benchmark_args


@is_mi300x
@pytest.mark.expensive
class BenchmarkLlama3_1_8B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/8b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path = self.weights_dir / "llama3.1_8b_instruct_fp16.irpa"
        self.irpa_path_fp8 = (
            self.artifacts_dir / "fp8/native_fp8_e4m3fnuz_llama3_8b.irpa"
        )
        self.irpa_path_fp8_attnf8 = (
            self.artifacts_dir / "fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa"
        )
        self.tensor_parallelism_size = 1
        self.dir_path_8b = self.dir_path / "llama-8b"
        self.temp_dir_8b = Path(self.dir_path_8b)
        self.temp_dir_8b.mkdir(parents=True, exist_ok=True)
        self.llama8b_f16_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            block_seq_stride=32,
        )
        self.llama8b_fp8_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            block_seq_stride=32,
            use_hf=True,
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
        )
        self.llama8b_fp8_attnf8_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8_attnf8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="sharktank",
            tensor_parallelism_size=self.tensor_parallelism_size,
            block_seq_stride=32,
            use_hf=True,
            activation_dtype="bfloat16",
            attention_dtype="float8_e4m3fnuz",
            kv_cache_dtype="float8_e4m3fnuz",
            use_attention_mask=True,
        )
        self.prefill_args_bs4_128_stride_32_f16 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp1"
        )
        self.decode_args_bs4_128_stride_32_f16 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32_tp1"
        )
        self.prefill_args_bs4_2048_stride_32_f16 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32"
        )
        self.decode_args_bs4_2048_stride_32_f16 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32"
        )
        # default fp8 input size here is 128
        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_nondecomposed_args_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_128_stride_32_f16,
        )
        self.iree_run_decode_nondecomposed_args_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_128_stride_32_f16,
        )
        self.iree_run_prefill_nondecomposed_args_fp16_2048 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_2048_stride_32_f16,
        )
        self.iree_run_decode_nondecomposed_args_fp16_2048 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_2048_stride_32_f16,
        )
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=4x128xi64=@{self.prefill_args_fp8}/tokens.bin",
            f"--input=4xi64=@{self.prefill_args_fp8}/seq_lens.bin",
            f"--input=4x4xi64=@{self.prefill_args_fp8}/seq_block_ids.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.prefill_args_fp8}/cs_f8E4M3FNUZ.bin",
            "--benchmark_repetitions=3",
            ">>",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=4x1xi64=@{self.decode_args_fp8}/next_tokens.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/seq_lens.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/start_positions.bin",
            f"--input=4x5xi64=@{self.decode_args_fp8}/seq_block_ids.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.decode_args_fp8}/cs_f8E4M3FNUZ.bin",
            "--benchmark_repetitions=3",
            ">>",
        ]
        self.iree_run_prefill_args_fp8_2048 = [
            "--function=prefill_bs4",
            f"--input=4x128xi64=@{self.prefill_args_fp8}/2048/prefill_token_ids_4x2048xi64.bin",
            f"--input=4xi64=@{self.prefill_args_fp8}/2048/prefill_seq_lens_4xi64.bin",
            f"--input=4x4xi64=@{self.prefill_args_fp8}/2048/prefill_seq_block_ids_4x64xi64.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.prefill_args_fp8}/2048/prefill_cache_state_261x2097152xf8E4M3FNUZ.bin",
            "--benchmark_repetitions=10",
            ">>",
        ]
        self.iree_run_decode_args_fp8_2048 = [
            "--function=decode_bs4",
            f"--input=4x1xi64=@{self.decode_args_fp8}/2048/decode_next_tokens_4x1xi64.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/2048/decode_seq_lens_4xi64.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/2048/decode_start_positions_4xi64.bin",
            f"--input=4x65xi64=@{self.decode_args_fp8}/2048/decode_seq_block_ids_tensor_4x65xi64.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.decode_args_fp8}/2048/decode_cache_state_261x2097152xf8E4M3FNUZ.bin",
            "--benchmark_repetitions=10",
            ">>",
        ]

    @skipif_run_quick_llama_test
    @pytest.mark.xfail(
        reason="Iree Compile Error", strict=True, raises=IreeCompileException
    )
    def testBenchmark8B_f16_TP1_Non_Decomposed_Input_Len_128(self):
        output_file_name = self.dir_path_8b / "f16_torch_128_tp1"
        output_mlir = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_benchmark = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".txt", prefix=output_file_name
        )
        export_return_code = self.llama8b_f16_torch_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_f16_torch_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama8b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_prefill_nondecomposed_args_fp16,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_decode_nondecomposed_args_fp16,
            cwd=self.repo_root,
        )

    @skipif_run_quick_llama_test
    @pytest.mark.xfail(
        reason="Iree Compile Error", strict=True, raises=IreeCompileException
    )
    def testBenchmark8B_f16_TP1_Non_Decomposed_Input_Len_2048(self):
        output_file_name = self.dir_path_8b / "f16_torch_2048_tp1"
        output_mlir = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_benchmark = self.llama8b_f16_torch_sdpa_artifacts.create_file(
            suffix=".txt", prefix=output_file_name
        )
        export_return_code = self.llama8b_f16_torch_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_f16_torch_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama8b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_prefill_nondecomposed_args_fp16_2048,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_f16_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_decode_nondecomposed_args_fp16_2048,
            cwd=self.repo_root,
        )

    @skipif_run_quick_llama_test
    @pytest.mark.xfail(reason="Benchmarking Error", raises=IreeBenchmarkException)
    def testBenchmark8B_fp8_TP1_Non_Decomposed(self):
        output_file_name = self.dir_path_8b / "fp8_torch_tp1"
        output_mlir = self.llama8b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_benchmark = self.llama8b_fp8_torch_sdpa_artifacts.create_file(
            suffix=".txt", prefix=output_file_name
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
            benchmark_filename=output_benchmark,
            args=self.iree_run_prefill_args_fp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_fp8_torch_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            benchmark_filename=output_benchmark,
            args=self.iree_run_decode_args_fp8,
            cwd=self.repo_root,
        )

    @skipif_run_quick_llama_test
    def testBenchmark8B_fp8_attnf8_TP1_Non_Decomposed_Input_Len_2048(self):
        output_file_name = self.dir_path_8b / "fp8_attnf8_2048_tp1"
        output_mlir = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_benchmark = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".txt", prefix=output_file_name
        )
        export_return_code = self.llama8b_fp8_attnf8_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_fp8_attnf8_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama8b_fp8_attnf8_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8_attnf8,
            benchmark_filename=output_benchmark,
            args=self.iree_run_prefill_args_fp8_2048,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_fp8_attnf8_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8_attnf8,
            benchmark_filename=output_benchmark,
            args=self.iree_run_decode_args_fp8_2048,
            cwd=self.repo_root,
        )

    @skipif_run_quick_llama_test
    def testBenchmark8B_fp8_attnf8_TP1_Non_Decomposed_Input_Len_128(self):
        output_file_name = self.dir_path_8b / "fp8_attnf8_128_tp1"
        output_mlir = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        output_benchmark = self.llama8b_fp8_attnf8_sdpa_artifacts.create_file(
            suffix=".txt", prefix=output_file_name
        )
        export_return_code = self.llama8b_fp8_attnf8_sdpa_artifacts.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama8b_fp8_attnf8_sdpa_artifacts.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama8b_fp8_attnf8_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8_attnf8,
            benchmark_filename=output_benchmark,
            args=self.iree_run_prefill_args_fp8,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama8b_fp8_attnf8_sdpa_artifacts.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8_attnf8,
            benchmark_filename=output_benchmark,
            args=self.iree_run_decode_args_fp8,
            cwd=self.repo_root,
        )


@is_mi300x
@pytest.mark.expensive
@skipif_run_quick_llama_test
class BenchmarkLlama3_1_70B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/70b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path = self.weights_dir / "llama3.1_70b_instruct_fp16.irpa"
        self.irpa_path_fp8 = self.artifacts_dir / "fp8/llama70b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.dir_path_70b = self.dir_path / "llama-70b"
        self.temp_dir_70b = Path(self.dir_path_70b)
        self.temp_dir_70b.mkdir(parents=True, exist_ok=True)
        self.llama70b_f16_torch_sdpa_artifacts_tp1 = ExportArtifacts(
            irpa_path=str(self.irpa_path),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            block_seq_stride=32,
        )
        self.llama70b_f16_torch_sdpa_artifacts_tp8 = ExportArtifacts(
            irpa_path=str(self.irpa_path),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            block_seq_stride=32,
        )
        self.llama70b_fp8_torch_sdpa_artifacts_tp1 = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            block_seq_stride=32,
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
        )
        self.prefill_args_bs4_128_stride_32_tp1_f16 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32"
        )
        self.decode_args_bs4_128_stride_32_tp1_f16 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32"
        )
        self.prefill_args_bs4_2048_stride_32_tp1_f16 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32"
        )
        self.decode_args_bs4_2048_stride_32_tp1_f16 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32"
        )
        self.prefill_args_bs4_128_stride_32_tp8_f16 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp8"
        )
        self.decode_args_bs4_128_stride_32_tp8_f16 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32_tp8"
        )
        self.prefill_args_bs4_2048_stride_32_tp8_f16 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32_tp8"
        )
        self.decode_args_bs4_2048_stride_32_tp8_f16 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32_tp8"
        )
        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_nondecomposed_args_128_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_128_stride_32_tp1_f16,
        )
        self.iree_run_decode_nondecomposed_args_128_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_128_stride_32_tp1_f16,
        )
        self.iree_run_prefill_nondecomposed_args_2048_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_2048_stride_32_tp1_f16,
        )
        self.iree_run_decode_nondecomposed_args_2048_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_2048_stride_32_tp1_f16,
        )
        self.iree_run_prefill_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_128_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_128_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_2048_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_2048_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
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

    def testBenchmark70B_f16_TP1_Non_Decomposed_Input_Len_128(self):
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
        output_benchmark = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".txt", prefix=output_file_name
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
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_prefill_nondecomposed_args_128_tp1_fp16,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_decode_nondecomposed_args_128_tp1_fp16,
            cwd=self.repo_root,
        )

    def testBenchmark70B_f16_TP1_Non_Decomposed_Input_Len_2048(self):
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
        output_benchmark = self.llama70b_f16_torch_sdpa_artifacts_tp1.create_file(
            suffix=".txt", prefix=output_file_name
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
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_prefill_nondecomposed_args_2048_tp1_fp16,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.iree_run_decode_nondecomposed_args_2048_tp1_fp16,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark70B_f16_TP8_Non_Decomposed_Input_Len_128(self):
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
            / f"tp8/llama3_70b_instruct_fp16_tp{self.tensor_parallelism_size}.irpa"
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
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_nondecomposed_args_128_tp8_fp16,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp8.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_nondecomposed_args_128_tp8_fp16,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark70B_f16_TP8_Non_Decomposed_Input_Len_2048(self):
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
            / f"tp8/llama3_70b_instruct_fp16_tp{self.tensor_parallelism_size}.irpa"
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
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_f16_torch_sdpa_artifacts_tp8.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            args=self.iree_run_decode_nondecomposed_args_2048_tp8_fp16,
            cwd=self.repo_root,
        )

    @pytest.mark.xfail(
        reason="70b fp8 irpa does not exist", strict=True, raises=ExportMlirException
    )
    def testBenchmark70B_fp8_TP1_Non_Decomposed(self):
        output_file_name = self.dir_path_70b / "fp8_torch_tp1"
        output_mlir = self.llama70b_fp8_torch_sdpa_artifacts_tp1.create_file(
            suffix=".mlir", prefix=output_file_name
        )
        output_json = self.llama70b_fp8_torch_sdpa_artifacts_tp1.create_file(
            suffix=".json", prefix=output_file_name
        )
        output_vmfb = self.llama70b_fp8_torch_sdpa_artifacts_tp1.create_file(
            suffix=".vmfb", prefix=output_file_name
        )
        export_return_code = self.llama70b_fp8_torch_sdpa_artifacts_tp1.export_to_mlir(
            mlir_path=output_mlir,
            json_path=output_json,
        )
        self.llama70b_fp8_torch_sdpa_artifacts_tp1.compile_to_vmfb(
            mlir_path=str(output_mlir),
            vmfb_path=output_vmfb,
            hal_dump_path=output_file_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        # benchmark prefill
        self.llama70b_fp8_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_prefill_args,
            cwd=self.repo_root,
        )
        # benchmark decode
        self.llama70b_fp8_torch_sdpa_artifacts_tp1.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path_fp8,
            args=self.iree_run_decode_args,
            cwd=self.repo_root,
        )


@is_mi300x
@pytest.mark.expensive
@skipif_run_quick_llama_test
class BenchmarkLlama3_1_405B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/405b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path = Path(
            "/shark-dev/data/llama3.1/weights/405b/fp16/llama3.1_405b_fp16.irpa"
        )
        self.irpa_path_fp8 = self.artifacts_dir / "f8/llama3.1_405b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.dir_path_405b = self.dir_path / "llama-405b"
        self.temp_dir_405b = Path(self.dir_path_405b)
        self.temp_dir_405b.mkdir(parents=True, exist_ok=True)
        self.llama405b_f16_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path),
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
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
        )
        self.prefill_args_bs4_128_stride_32_tp8_f16 = (
            self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp8"
        )
        self.decode_args_bs4_128_stride_32_tp8_f16 = (
            self.artifacts_dir / "decode_args_bs4_128_stride_32_tp8"
        )
        self.prefill_args_bs4_2048_stride_32_tp8_f16 = (
            self.artifacts_dir / "prefill_args_bs4_2048_stride_32_tp8"
        )
        self.decode_args_bs4_2048_stride_32_tp8_f16 = (
            self.artifacts_dir / "decode_args_bs4_2048_stride_32_tp8"
        )
        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
        self.iree_run_prefill_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_128_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_128_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.prefill_args_bs4_2048_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.decode_args_bs4_2048_stride_32_tp8_f16,
            tensor_parallelism_size=self.tensor_parallelism_size,
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
    def testBenchmark405B_f16_TP8_Non_Decomposed_Input_Len_128(self):
        output_file_name = self.dir_path_405b / "f16_torch_128"
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
            self.artifacts_dir
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
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_nondecomposed_args_128_tp8_fp16,
            cwd=self.repo_root,
        )
        # TODO: benchmark decode

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark405B_f16_TP8_Non_Decomposed_Input_Len_2048(self):
        output_file_name = self.dir_path_405b / "f16_torch_2048"
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
            self.artifacts_dir
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
            irpa_path=self.irpa_path,
            args=self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16,
            cwd=self.repo_root,
        )
        # TODO: benchmark decode

    @pytest.mark.xfail(
        reason="KeyError in theta.py", strict=True, raises=ExportMlirException
    )
    def testBenchmark405B_fp8_TP8_Non_Decomposed(self):
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
