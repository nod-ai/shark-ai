# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import datetime
import itertools
import os
import unittest
from parameterized import parameterized
import pytest
from pathlib import Path
from sharktank.utils.export_artifacts import (
    ExportArtifacts,
    ExportMlirException,
    IreeBenchmarkException,
)
from sharktank.utils.testing import (
    is_llama_8b,
    is_mi300x,
    is_mi350x,
    is_nightly,
)
from ireers_tools import *
import json


@pytest.mark.usefixtures("iree_flags", "model_artifacts")
class BaseBenchmarkTest(unittest.TestCase):
    dir_path_suffix = datetime.now().strftime("%Y-%m-%d")
    repo_root = Path(__file__).resolve().parents[4]
    dir_path = repo_root / dir_path_suffix

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.dir_path, exist_ok=True)

    def setUp(self, artifact_dir: Path, dir_path_name: str):
        super().setUp()
        self.export_artifact: ExportArtifacts
        self.compile_args = [
            "--iree-opt-level=O3",
            "--iree-hal-indirect-command-buffers=true",
            "--iree-stream-resource-memory-model=discrete",
            "--iree-hal-memoization=true",
            "--iree-stream-affinity-solver-max-iterations=1024",
        ]
        self.artifact_dir = artifact_dir
        self.dir_path = self.__class__.dir_path / dir_path_name
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)

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
        self.export_artifact.export_and_compile_llm(
            batch_size=self.batch_size, skip_decode=skip_decode
        )

        benchmark_filename = self.export_artifact.output_name.with_suffix(".txt")
        self.export_artifact.iree_benchmark(
            benchmark_filename=benchmark_filename,
            extra_args=self.prefill_args,
        )
        if not skip_decode:
            self.export_artifact.iree_benchmark(
                benchmark_filename=benchmark_filename,
                extra_args=self.decode_args,
            )

    def fetch_source_fixtures_for_run_flags(self, inference_list, model_name, submodel_name):
        result = []
        for entry in inference_list:
            source = entry.get("source")
            value = entry.get("value")
            source_fixture = fetch_source_fixture(
                source, group=f"{model_name}_{submodel_name}"
            )
            result.append([source_fixture.path, value])

        return result


    def common_run_flags_generation(self, input_list, output_list):
        flags_list = []

        if input_list:
            for path, value in input_list:
                if not value:
                    flags_list.append(f"--input=@{path}")
                else:
                    flags_list.append(f"--input={value}=@{path}")

        if output_list:
            for path, value in output_list:
                if not value:
                    flags_list.append(f"--expected_output=@{path}")
                else:
                    flags_list.append(f"--expected_output={value}=@{path}")

        return flags_list

    def export_compile_run(self, skip_decode: bool = False):
        self.export_artifact.export_and_compile_llm(
            batch_size=self.batch_size, skip_decode=skip_decode
        )

        self.export_artifact.iree_run(
            extra_args=self.prefill_args,
        )
        if not skip_decode:
            self.export_artifact.iree_run(
                extra_args=self.decode_args,
            )


@is_mi300x
class BenchmarkLlama3_1_8B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp(artifact_dir=Path("/shark-dev/8b"), dir_path_name="llama-8b")
        # TODO: add numpy files to Azure and download from it
        self.batch_size = 4

        self.prefill_args_fp16 = {
            128: self.save_benchmarks(
                benchmark_fn="prefill_bs4",
                input_path=self.artifact_dir / "prefill_args_bs4_128_stride_32_tp1",
                tensor_parallelism_size=1,
            ),
            2048: self.save_benchmarks(
                benchmark_fn="prefill_bs4",
                input_path=self.artifact_dir / "prefill_args_bs4_2048_stride_32",
                tensor_parallelism_size=1,
            ),
        }

        self.decode_args_fp16 = {
            128: self.save_benchmarks(
                benchmark_fn="decode_bs4",
                input_path=self.artifact_dir / "decode_args_bs4_128_stride_32_tp1",
                tensor_parallelism_size=1,
            ),
            2048: self.save_benchmarks(
                benchmark_fn="decode_bs4",
                input_path=self.artifact_dir / "decode_args_bs4_2048_stride_32",
                tensor_parallelism_size=1,
            ),
        }

        # default fp8 input size here is 128
        prefill_args_fp8_path = self.artifact_dir / "prefill_args_fp8"
        decode_args_fp8_path = self.artifact_dir / "decode_args_fp8"
        self.prefill_args_fp8 = {
            128: [
                "--function=prefill_bs4",
                f"--input=4x128xi64=@{prefill_args_fp8_path}/tokens.bin",
                f"--input=4xi64=@{prefill_args_fp8_path}/seq_lens.bin",
                f"--input=4x4xi64=@{prefill_args_fp8_path}/seq_block_ids.bin",
                f"--input=261x2097152xf8E4M3FNUZ=@{prefill_args_fp8_path}/cs_f8E4M3FNUZ.bin",
                "--benchmark_repetitions=10",
                ">>",
            ],
            2048: [
                "--function=prefill_bs4",
                f"--input=4x2048xi64=@{prefill_args_fp8_path}/2048/prefill_token_ids_4x2048xi64.bin",
                f"--input=4xi64=@{prefill_args_fp8_path}/2048/prefill_seq_lens_4xi64.bin",
                f"--input=4x64xi64=@{prefill_args_fp8_path}/2048/prefill_seq_block_ids_4x64xi64.bin",
                f"--input=261x2097152xf8E4M3FNUZ=@{prefill_args_fp8_path}/2048/prefill_cache_state_261x2097152xf8E4M3FNUZ.bin",
                "--benchmark_repetitions=10",
                ">>",
            ],
        }

        self.decode_args_fp8 = {
            128: [
                "--function=decode_bs4",
                f"--input=4x1xi64=@{decode_args_fp8_path}/next_tokens.bin",
                f"--input=4xi64=@{decode_args_fp8_path}/seq_lens.bin",
                f"--input=4xi64=@{decode_args_fp8_path}/start_positions.bin",
                f"--input=4x5xi64=@{decode_args_fp8_path}/seq_block_ids.bin",
                f"--input=261x2097152xf8E4M3FNUZ=@{decode_args_fp8_path}/cs_f8E4M3FNUZ.bin",
                "--benchmark_repetitions=10",
                ">>",
            ],
            2048: [
                "--function=decode_bs4",
                f"--input=4x1xi64=@{decode_args_fp8_path}/2048/decode_next_tokens_4x1xi64.bin",
                f"--input=4xi64=@{decode_args_fp8_path}/2048/decode_seq_lens_4xi64.bin",
                f"--input=4xi64=@{decode_args_fp8_path}/2048/decode_start_positions_4xi64.bin",
                f"--input=4x65xi64=@{decode_args_fp8_path}/2048/decode_seq_block_ids_tensor_4x65xi64.bin",
                f"--input=261x2097152xf8E4M3FNUZ=@{decode_args_fp8_path}/2048/decode_cache_state_261x2097152xf8E4M3FNUZ.bin",
                "--benchmark_repetitions=10",
                ">>",
            ],
        }

    @is_llama_8b
    def test_benchmark8B_f16_tp1_input_len_128(self):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.llama3_8b_f16_model,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            output_name=self.dir_path / f"f16_torch_{128}_tp1",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.prefill_args_fp16[128]
        self.decode_args = self.decode_args_fp16[128]

        self.export_compile_benchmark()

    @is_nightly
    def test_benchmark8B_f16_tp1_input_len_2048(self):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.llama3_8b_f16_model,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            output_name=self.dir_path / f"f16_torch_{2048}_tp1",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.prefill_args_fp16[2048]
        self.decode_args = self.decode_args_fp16[2048]

        self.export_compile_benchmark()

    @is_nightly
    def test_benchmark8B_fp8_tp1_input_len_128(self):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.llama3_8b_f8_model,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            use_hf=True,
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
            output_name=self.dir_path / "fp8_torch_tp1",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.prefill_args_fp8[128]
        self.decode_args = self.decode_args_fp8[128]

        self.export_compile_benchmark()

    @parameterized.expand((((128,), (2048,))))
    @is_nightly
    def test_benchmark8B_fp8_attnf8_tp1(self, input_size: int):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.llama3_8b_f8_attnf8_model,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="sharktank",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            use_hf=True,
            activation_dtype="bfloat16",
            attention_dtype="float8_e4m3fnuz",
            kv_cache_dtype="float8_e4m3fnuz",
            use_attention_mask=True,
            output_name=self.dir_path / f"fp8_attnf8_{input_size}_tp1",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.prefill_args_fp8[input_size]
        self.decode_args = self.decode_args_fp8[input_size]

        self.export_compile_benchmark()


@is_mi300x
@is_nightly
class BenchmarkLlama3_1_70B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp(artifact_dir=Path("/shark-dev/70b"), dir_path_name="llama-70b")
        # TODO: add numpy files to Azure and download from it

        self.batch_size = 4

        self.prefill_args_fp16 = {
            1: {
                128: self.save_benchmarks(
                    benchmark_fn="prefill_bs4",
                    input_path=self.artifact_dir / "prefill_args_bs4_128_stride_32",
                    tensor_parallelism_size=1,
                ),
                2048: self.save_benchmarks(
                    benchmark_fn="prefill_bs4",
                    input_path=self.artifact_dir / "prefill_args_bs4_2048_stride_32",
                    tensor_parallelism_size=1,
                ),
            },
            8: {
                128: self.save_benchmarks(
                    benchmark_fn="prefill_bs4",
                    input_path=self.artifact_dir / "prefill_args_bs4_128_stride_32_tp8",
                    tensor_parallelism_size=8,
                ),
                2048: self.save_benchmarks(
                    benchmark_fn="prefill_bs4",
                    input_path=self.artifact_dir
                    / "prefill_args_bs4_2048_stride_32_tp8",
                    tensor_parallelism_size=8,
                ),
            },
        }

        self.decode_args_fp16 = {
            1: {
                128: self.save_benchmarks(
                    benchmark_fn="decode_bs4",
                    input_path=self.artifact_dir / "decode_args_bs4_128_stride_32",
                    tensor_parallelism_size=1,
                ),
                2048: self.save_benchmarks(
                    benchmark_fn="decode_bs4",
                    input_path=self.artifact_dir / "decode_args_bs4_2048_stride_32",
                    tensor_parallelism_size=1,
                ),
            },
            8: {
                128: self.save_benchmarks(
                    benchmark_fn="decode_bs4",
                    input_path=self.artifact_dir / "decode_args_bs4_128_stride_32_tp8",
                    tensor_parallelism_size=8,
                ),
                2048: self.save_benchmarks(
                    benchmark_fn="decode_bs4",
                    input_path=self.artifact_dir / "decode_args_bs4_2048_stride_32_tp8",
                    tensor_parallelism_size=8,
                ),
            },
        }

        prefill_args_fp8_path = self.artifact_dir / "prefill_args_fp8"
        decode_args_fp8_path = self.artifact_dir / "decode_args_fp8"
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{prefill_args_fp8_path}/tokens.npy",
            f"--input=@{prefill_args_fp8_path}/seq_lens.npy",
            f"--input=@{prefill_args_fp8_path}/seq_block_ids.npy",
            f"--input=@{prefill_args_fp8_path}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{decode_args_fp8_path}/tokens.npy",
            f"--input=@{decode_args_fp8_path}/seq_lens.npy",
            f"--input=@{decode_args_fp8_path}/start_positions.npy",
            f"--input=@{decode_args_fp8_path}/seq_block_ids.npy",
            f"--input=@{decode_args_fp8_path}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    @parameterized.expand(tuple(itertools.product((128, 2048), (1, 8))))
    @pytest.mark.xfail(
        reason="https://github.com/nod-ai/shark-ai/issues/1355",
        strict=False,
        raises=IreeBenchmarkException,
    )
    def test_benchmark70B_f16(self, input_size: int, tp: int):
        output_name = self.dir_path / f"f16_torch_{input_size}_tp{tp}"
        if tp == 1:
            irpa_path = self.llama3_70b_f16_model
        else:
            assert tp == 8
            irpa_path = self.llama3_70b_f16_tp8_model
        self.export_artifact = ExportArtifacts(
            irpa_path=irpa_path,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=tp,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            output_name=output_name,
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.prefill_args_fp16[tp][input_size]
        self.decode_args = self.decode_args_fp16[tp][input_size]

        self.export_compile_benchmark()

    @pytest.mark.xfail(
        reason="70b fp8 irpa does not exist", strict=True, raises=ExportMlirException
    )
    def test_benchmark70B_fp8_tp1(self):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.llama3_70b_f8_model,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
            output_name=self.dir_path / "fp8_torch_tp1",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.iree_run_prefill_args_fp8
        self.decode_args = self.iree_run_decode_args_fp8

        self.export_compile_benchmark()


@is_mi300x
@is_nightly
class BenchmarkLlama3_1_405B(BaseBenchmarkTest):
    def setUp(self):
        super().setUp(artifact_dir=Path("/shark-dev/405b"), dir_path_name="llama-405b")
        # TODO: add numpy files to Azure and download from it

        self.batch_size = 4

        self.prefill_args_tp8_fp16 = {
            128: self.save_benchmarks(
                benchmark_fn="prefill_bs4",
                input_path=self.artifact_dir / "prefill_args_bs4_128_stride_32_tp8",
                tensor_parallelism_size=8,
            ),
            2048: self.save_benchmarks(
                benchmark_fn="prefill_bs4",
                input_path=self.artifact_dir / "prefill_args_bs4_2048_stride_32_tp8",
                tensor_parallelism_size=8,
            ),
        }
        self.decode_args_tp8_fp16 = {
            128: self.save_benchmarks(
                benchmark_fn="decode_bs4",
                input_path=self.artifact_dir / "decode_args_bs4_128_stride_32_tp8",
                tensor_parallelism_size=8,
            ),
            2048: self.save_benchmarks(
                benchmark_fn="decode_bs4",
                input_path=self.artifact_dir / "decode_args_bs4_2048_stride_32_tp8",
                tensor_parallelism_size=8,
            ),
        }

        prefill_args_fp8_path = self.artifact_dir / "prefill_args_fp8"
        decode_args_fp8_path = self.artifact_dir / "decode_args_fp8"
        self.iree_run_prefill_args_fp8 = [
            "--function=prefill_bs4",
            f"--input=@{prefill_args_fp8_path}/tokens.npy",
            f"--input=@{prefill_args_fp8_path}/seq_lens.npy",
            f"--input=@{prefill_args_fp8_path}/seq_block_ids.npy",
            f"--input=@{prefill_args_fp8_path}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]
        self.iree_run_decode_args_fp8 = [
            "--function=decode_bs4",
            f"--input=@{decode_args_fp8_path}/tokens.npy",
            f"--input=@{decode_args_fp8_path}/seq_lens.npy",
            f"--input=@{decode_args_fp8_path}/start_positions.npy",
            f"--input=@{decode_args_fp8_path}/seq_block_ids.npy",
            f"--input=@{decode_args_fp8_path}/cache_state_f16.npy",
            "--benchmark_repetitions=3",
        ]

    @parameterized.expand((((128,), (2048,))))
    @pytest.mark.xfail(
        reason="Benchmarking Error", strict=True, raises=IreeBenchmarkException
    )
    def test_benchmark405B_f16_tp8(self, input_size: int):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.llama3_405b_f16_tp8_model,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=8,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            output_name=self.dir_path / f"f16_torch_{input_size}",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.prefill_args_tp8_fp16[input_size]
        self.decode_args = self.decode_args_tp8_fp16[input_size]

        self.export_compile_benchmark(skip_decode=True)  # TODO: Enable decode

    @pytest.mark.xfail(
        reason="KeyError in theta.py", strict=True, raises=ExportMlirException
    )
    def test_benchmark405B_fp8_tp8(self):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.llama3_405b_f8_tp8_model,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=8,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            activation_dtype="bfloat16",
            attention_dtype="bfloat16",
            kv_cache_dtype="float8_e4m3fnuz",
            output_name=self.dir_path / "fp8_torch",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.iree_run_prefill_args_fp8
        self.decode_args = self.iree_run_decode_args_fp8

        self.export_compile_benchmark(skip_decode=True)  # TODO: Enable decode


# TODO: Add mi350 runner to the shark-ai CI 
@is_mi350x
@is_nightly
class BenchmarkLlama3_1_405B_fp4(BaseBenchmarkTest):
    def setUp(self):
        self.batch_size = 4
        self.model_name = "llama"
        self.submodel_name = "405b_fp4"
        self.file_path = Path(__file__).parent / "405b_fp4_gemm_f16_kv_cache.json"
        with open(self.file_path, "r") as file:
            data = json.load(file)

            # retrieving source fixtures if available in JSON file
            self.inputs = (
                self.fetch_source_fixtures_for_run_flags(
                    data.get("inputs"), self.model_name, self.submodel_name
                )
                if data.get("inputs")
                else None
            )
            self.outputs = (
                self.fetch_source_fixtures_for_run_flags(
                    data.get("outputs"), self.model_name, self.submodel_name
                )
                if data.get("outputs")
                else None
            )
            real_weights_group_name = "llama3_405b_instruct_mi355_fp4_2025_07_10_fn"
            self.real_weights = (
                fetch_source_fixture(
                    data.get("real_weights"),
                    group=real_weights_group_name,
                )
                if data.get("real_weights")
                else None
            )

            self.common_rule_flags = self.common_run_flags_generation(
                self.inputs, self.outputs
            )
            self.run_function = data.get("run_function")

        self.prefill_args_tp1_fp4 = {
            32: self.common_rule_flags
        }


    @parameterized.expand((((32,))))
    def test_benchmark405B_fp4_tp1(self, input_size: int):
        self.export_artifact = ExportArtifacts(
            irpa_path=self.real_weights,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            attention_kernel="torch",
            tensor_parallelism_size=1,
            pipeline_parallelism_size=1,
            block_seq_stride=32,
            cwd=self.repo_root,
            output_name=self.dir_path / f"fp4_torch_{input_size}",
            hip_device_id=self.iree_device,
        )
        self.prefill_args = self.prefill_args_tp1_fp4[input_size]

        self.export_compile_run(skip_decode=True)  # TODO: Enable decode


if __name__ == "__main__":
    unittest.main()
