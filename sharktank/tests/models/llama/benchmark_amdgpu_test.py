# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import datetime
import os
import unittest
import pytest
from pathlib import Path
from sharktank.utils.export_artifacts import (
    ExportArtifacts,
    ExportMlirException,
    IreeBenchmarkException,
)
from sharktank.utils.testing import (
    is_mi300x,
    is_nightly,
)


@pytest.mark.usefixtures("get_iree_flags")
class BaseBenchmarkTest(unittest.TestCase):
    dir_path_suffix = datetime.now().strftime("%Y-%m-%d")
    repo_root = Path(__file__).resolve().parents[4]
    dir_path = repo_root / dir_path_suffix

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.dir_path, exist_ok=True)

    def setUp(self):
        super().setUp()
        self.compile_args = [
            "--iree-opt-level=O3",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hal-memoization=true",
            "--iree-stream-affinity-solver-max-iterations=1024",
        ]
        self.tensor_parallelism_size = 1
        self.pipeline_parallelism_size = 1

    def save_benchmarks(
        self,
        *,
        benchmark_fn: str,
        input_path: Path,
        tensor_parallelism_size: int = 1,
        benchmark_repetitions: int = 3,
    ) -> list[str]:
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

        # TODO: Support pipeline parallelism
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

    def export_compile_benchmark(self, skip_decode: bool = False):
        output_mlir = self.export_artifact.create_file(
            prefix=self.output_name, suffix=".mlir"
        )
        output_json = self.export_artifact.create_file(
            prefix=self.output_name, suffix=".json"
        )
        output_vmfb = self.export_artifact.create_file(
            prefix=self.output_name, suffix=".vmfb"
        )
        output_benchmark = self.export_artifact.create_file(
            prefix=self.output_name, suffix=".txt"
        )
        self.export_artifact.export_to_mlir(
            output_mlir=output_mlir,
            output_config=output_json,
        )
        self.export_artifact.compile_to_vmfb(
            output_mlir=str(output_mlir),
            output_vmfb=output_vmfb,
            hal_dump_path=self.output_name,
            cwd=self.repo_root,
            args=self.compile_args,
        )
        self.export_artifact.iree_benchmark_vmfb(
            hip_device_id=self.iree_device,
            vmfb_name=output_vmfb,
            irpa_path=self.irpa_path,
            benchmark_filename=output_benchmark,
            args=self.prefill_args,
            cwd=self.repo_root,
        )
        if not skip_decode:
            self.export_artifact.iree_benchmark_vmfb(
                hip_device_id=self.iree_device,
                vmfb_name=output_vmfb,
                irpa_path=self.irpa_path,
                benchmark_filename=output_benchmark,
                args=self.decode_args,
                cwd=self.repo_root,
            )


@is_mi300x
class BenchmarkLlama3_1_8B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/8b")
        self.irpa_path_fp16 = (
            self.artifacts_dir / "instruct/weights/llama3.1_8b_instruct_fp16.irpa"
        )
        self.irpa_path_fp8 = (
            self.artifacts_dir / "fp8/native_fp8_e4m3fnuz_llama3_8b.irpa"
        )
        self.irpa_path_fp8_attnf8 = (
            self.artifacts_dir / "fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa"
        )
        self.dir_path = self.__class__.dir_path / "llama-8b"
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)

        self.llama8b_f16_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp16),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            pipeline_parallelism_size=self.pipeline_parallelism_size,
            block_seq_stride=32,
        )
        self.llama8b_fp8_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            pipeline_parallelism_size=self.pipeline_parallelism_size,
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
            pipeline_parallelism_size=self.pipeline_parallelism_size,
            block_seq_stride=32,
            use_hf=True,
            activation_dtype="bfloat16",
            attention_dtype="float8_e4m3fnuz",
            kv_cache_dtype="float8_e4m3fnuz",
            use_attention_mask=True,
        )

        # default fp8 input size here is 128
        self.prefill_args_nondecomposed_fp16_128 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp1",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.decode_args_nondecomposed_fp16_128 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_128_stride_32_tp1",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.prefill_args_nondecomposed_fp16_2048 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_2048_stride_32",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.decode_args_nondecomposed_fp16_2048 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_2048_stride_32",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )

        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
        self.prefill_args_nondecomposed_fp8_128 = [
            "--function=prefill_bs4",
            f"--input=4x128xi64=@{self.prefill_args_fp8}/tokens.bin",
            f"--input=4xi64=@{self.prefill_args_fp8}/seq_lens.bin",
            f"--input=4x4xi64=@{self.prefill_args_fp8}/seq_block_ids.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.prefill_args_fp8}/cs_f8E4M3FNUZ.bin",
            "--benchmark_repetitions=10",
            ">>",
        ]
        self.decode_args_nondecomposed_fp8_128 = [
            "--function=decode_bs4",
            f"--input=4x1xi64=@{self.decode_args_fp8}/next_tokens.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/seq_lens.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/start_positions.bin",
            f"--input=4x5xi64=@{self.decode_args_fp8}/seq_block_ids.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.decode_args_fp8}/cs_f8E4M3FNUZ.bin",
            "--benchmark_repetitions=10",
            ">>",
        ]
        self.prefill_args_nondecomposed_fp8_2048 = [
            "--function=prefill_bs4",
            f"--input=4x2048xi64=@{self.prefill_args_fp8}/2048/prefill_token_ids_4x2048xi64.bin",
            f"--input=4xi64=@{self.prefill_args_fp8}/2048/prefill_seq_lens_4xi64.bin",
            f"--input=4x64xi64=@{self.prefill_args_fp8}/2048/prefill_seq_block_ids_4x64xi64.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.prefill_args_fp8}/2048/prefill_cache_state_261x2097152xf8E4M3FNUZ.bin",
            "--benchmark_repetitions=10",
            ">>",
        ]
        self.decode_args_nondecomposed_fp8_2048 = [
            "--function=decode_bs4",
            f"--input=4x1xi64=@{self.decode_args_fp8}/2048/decode_next_tokens_4x1xi64.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/2048/decode_seq_lens_4xi64.bin",
            f"--input=4xi64=@{self.decode_args_fp8}/2048/decode_start_positions_4xi64.bin",
            f"--input=4x65xi64=@{self.decode_args_fp8}/2048/decode_seq_block_ids_tensor_4x65xi64.bin",
            f"--input=261x2097152xf8E4M3FNUZ=@{self.decode_args_fp8}/2048/decode_cache_state_261x2097152xf8E4M3FNUZ.bin",
            "--benchmark_repetitions=10",
            ">>",
        ]

    def testBenchmark8B_f16_TP1_Non_Decomposed_Input_Len_128(self):
        self.output_name = self.dir_path / "f16_torch_128_tp1"
        self.export_artifact = self.llama8b_f16_torch_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.prefill_args_nondecomposed_fp16_128
        self.decode_args = self.decode_args_nondecomposed_fp16_128

        self.export_compile_benchmark()

    @is_nightly
    def testBenchmark8B_f16_TP1_Non_Decomposed_Input_Len_2048(self):
        self.output_name = self.dir_path / "f16_torch_2048_tp1"
        self.export_artifact = self.llama8b_f16_torch_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.prefill_args_nondecomposed_fp16_2048
        self.decode_args = self.decode_args_nondecomposed_fp16_2048

        self.export_compile_benchmark()

    @is_nightly
    def testBenchmark8B_fp8_TP1_Non_Decomposed_Input_len_128(self):
        self.output_name = self.dir_path / "fp8_torch_tp1"
        self.export_artifact = self.llama8b_fp8_torch_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp8
        self.prefill_args = self.prefill_args_nondecomposed_fp8_128
        self.decode_args = self.decode_args_nondecomposed_fp8_128

        self.export_compile_benchmark()

    @is_nightly
    def testBenchmark8B_fp8_attnf8_TP1_Non_Decomposed_Input_Len_2048(self):
        self.output_name = self.dir_path / "fp8_attnf8_2048_tp1"
        self.export_artifact = self.llama8b_fp8_attnf8_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp8_attnf8
        self.prefill_args = self.prefill_args_nondecomposed_fp8_2048
        self.decode_args = self.decode_args_nondecomposed_fp8_2048

        self.export_compile_benchmark()

    @is_nightly
    def testBenchmark8B_fp8_attnf8_TP1_Non_Decomposed_Input_Len_128(self):
        self.output_name = self.dir_path / "fp8_attnf8_128_tp1"
        self.export_artifact = self.llama8b_fp8_attnf8_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp8_attnf8
        self.prefill_args = self.prefill_args_nondecomposed_fp8_128
        self.decode_args = self.decode_args_nondecomposed_fp8_128

        self.export_compile_benchmark()


@is_mi300x
@is_nightly
class BenchmarkLlama3_1_70B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/70b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path_fp16 = self.weights_dir / "llama3.1_70b_instruct_fp16.irpa"
        self.irpa_path_fp8 = self.artifacts_dir / "fp8/llama70b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.dir_path = self.__class__.dir_path / "llama-70b"
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)
        self.llama70b_f16_torch_sdpa_artifacts_tp1 = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp16),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
        )
        self.llama70b_f16_torch_sdpa_artifacts_tp8 = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp16),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
        )
        self.llama70b_fp8_torch_sdpa_artifacts_tp1 = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
        )

        self.iree_run_prefill_nondecomposed_args_128_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_128_stride_32",
            tensor_parallelism_size=1,
        )
        self.iree_run_decode_nondecomposed_args_128_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_128_stride_32",
            tensor_parallelism_size=1,
        )
        self.iree_run_prefill_nondecomposed_args_2048_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_2048_stride_32",
            tensor_parallelism_size=1,
        )
        self.iree_run_decode_nondecomposed_args_2048_tp1_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_2048_stride_32",
            tensor_parallelism_size=1,
        )
        self.iree_run_prefill_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_128_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_2048_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_2048_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )

        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
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
        self.output_name = self.dir_path / "f16_torch_128_tp1"
        self.export_artifact = self.llama70b_f16_torch_sdpa_artifacts_tp1
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.iree_run_prefill_nondecomposed_args_128_tp1_fp16
        self.decode_args = self.iree_run_decode_nondecomposed_args_128_tp1_fp16

        self.export_compile_benchmark()

    def testBenchmark70B_f16_TP1_Non_Decomposed_Input_Len_2048(self):
        self.output_name = self.dir_path / "f16_torch_2048_tp1"
        self.export_artifact = self.llama70b_f16_torch_sdpa_artifacts_tp1
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.iree_run_prefill_nondecomposed_args_2048_tp1_fp16
        self.decode_args = self.iree_run_decode_nondecomposed_args_2048_tp1_fp16

        self.export_compile_benchmark()

    @pytest.mark.xfail(
        reason="https://github.com/nod-ai/shark-ai/issues/1355",
        strict=True,
        raises=IreeBenchmarkException,
    )
    def testBenchmark70B_f16_TP8_Non_Decomposed_Input_Len_128(self):
        self.output_name = self.dir_path / "f16_torch_128_tp8"
        self.export_artifact = self.llama70b_f16_torch_sdpa_artifacts_tp8
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.iree_run_prefill_nondecomposed_args_128_tp8_fp16
        self.decode_args = self.iree_run_decode_nondecomposed_args_128_tp8_fp16

        self.export_compile_benchmark()

    @pytest.mark.xfail(
        reason="https://github.com/nod-ai/shark-ai/issues/1355",
        strict=True,
        raises=IreeBenchmarkException,
    )
    def testBenchmark70B_f16_TP8_Non_Decomposed_Input_Len_2048(self):
        self.output_name = self.dir_path / "f16_torch_2048_tp8"
        self.export_artifact = self.llama70b_f16_torch_sdpa_artifacts_tp8
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16
        self.decode_args = self.iree_run_decode_nondecomposed_args_2048_tp8_fp16

        self.export_compile_benchmark()

    @pytest.mark.xfail(
        reason="70b fp8 irpa does not exist", strict=True, raises=ExportMlirException
    )
    def testBenchmark70B_fp8_TP1_Non_Decomposed(self):
        self.output_name = self.dir_path / "fp8_torch_tp1"
        self.export_artifact = self.llama70b_fp8_torch_sdpa_artifacts_tp1
        self.irpa_path = self.irpa_path_fp8
        self.prefill_args = self.iree_run_prefill_args_fp8
        self.decode_args = self.iree_run_decode_args_fp8

        self.export_compile_benchmark()


@is_mi300x
@is_nightly
class BenchmarkLlama3_1_405B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp()
        # TODO: add numpy files to Azure and download from it
        self.artifacts_dir = Path("/shark-dev/405b")
        self.weights_dir = self.artifacts_dir / "instruct/weights"
        self.irpa_path_fp16 = Path(
            "/shark-dev/data/llama3.1/weights/405b/fp16/llama3.1_405b_fp16.irpa"
        )
        self.irpa_path_fp8 = self.artifacts_dir / "f8/llama3.1_405b_fp8.irpa"
        self.tensor_parallelism_size = 8
        self.dir_path = self.__class__.dir_path / "llama-405b"
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)

        self.llama405b_f16_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp16),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            pipeline_parallelism_size=self.pipeline_parallelism_size,
            block_seq_stride=32,
        )
        self.llama405b_fp8_torch_sdpa_artifacts = ExportArtifacts(
            irpa_path=str(self.irpa_path_fp8),
            batch_size=4,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            attention_kernel="torch",
            tensor_parallelism_size=self.tensor_parallelism_size,
            pipeline_parallelism_size=self.pipeline_parallelism_size,
            block_seq_stride=32,
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
        )

        self.iree_run_prefill_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_128_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_128_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_128_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="prefill_bs4",
            input_path=self.artifacts_dir / "prefill_args_bs4_2048_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )
        self.iree_run_decode_nondecomposed_args_2048_tp8_fp16 = self.save_benchmarks(
            benchmark_fn="decode_bs4",
            input_path=self.artifacts_dir / "decode_args_bs4_2048_stride_32_tp8",
            tensor_parallelism_size=self.tensor_parallelism_size,
        )

        self.prefill_args_fp8 = self.artifacts_dir / "prefill_args_fp8"
        self.decode_args_fp8 = self.artifacts_dir / "decode_args_fp8"
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
        self.output_name = self.dir_path / "f16_torch_128"
        self.export_artifact = self.llama405b_f16_torch_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.iree_run_prefill_nondecomposed_args_128_tp8_fp16
        self.decode_args = self.iree_run_decode_nondecomposed_args_128_tp8_fp16

        self.export_compile_benchmark(skip_decode=True)  # TODO: Enable decode

    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def testBenchmark405B_f16_TP8_Non_Decomposed_Input_Len_2048(self):
        self.output_name = self.dir_path / "f16_torch_2048"
        self.export_artifact = self.llama405b_f16_torch_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp16
        self.prefill_args = self.iree_run_prefill_nondecomposed_args_2048_tp8_fp16
        self.decode_args = self.iree_run_decode_nondecomposed_args_2048_tp8_fp16

        self.export_compile_benchmark(skip_decode=True)  # TODO: Enable decode

    @pytest.mark.xfail(
        reason="KeyError in theta.py", strict=True, raises=ExportMlirException
    )
    def testBenchmark405B_fp8_TP8_Non_Decomposed(self):
        self.output_name = self.dir_path / "fp8_torch"
        self.export_artifact = self.llama405b_fp8_torch_sdpa_artifacts
        self.irpa_path = self.irpa_path_fp8
        self.prefill_args = self.iree_run_prefill_args_fp8
        self.decode_args = self.iree_run_decode_args_fp8

        self.export_compile_benchmark(skip_decode=True)  # TODO: Enable decode


if __name__ == "__main__":
    unittest.main()
