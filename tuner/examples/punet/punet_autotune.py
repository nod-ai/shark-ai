# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Sample Usage:

python -m examples.punet benchmark.mlir --lhs-dims=bmk --rhs-dims=bkn --tile-dims=*mnk --devices=hip://0,hip://1 --num-candidates=64


Recommended Trial Run:

python -m examples.punet benchmark.mlir --num-candidates=1


Dry Run Test (no gpu requried):

python -m examples.punet benchmark.mlir --num-candidates=64 --num-model-candidates=10 --dry-run

"""

from tuner import libtuner
from pathlib import Path


class PunetClient(libtuner.TuningClient):
    def get_dispatch_compile_timeout_s(self) -> int:
        return 4

    def get_dispatch_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        mlir_path = candidate_tracker.dispatch_mlir_path
        assert mlir_path is not None
        command = [
            "compile_candidate.sh",
            mlir_path.as_posix(),
        ]
        return command

    def get_dispatch_benchmark_timeout_s(self) -> int:
        return 15

    def get_dispatch_benchmark_command(
        self,
        candidate_tracker: libtuner.CandidateTracker,
    ) -> list[str]:
        compiled_vmfb_path = candidate_tracker.compiled_dispatch_path
        assert compiled_vmfb_path is not None

        command = [
            "iree-benchmark-module",
            f"--device={libtuner.DEVICE_ID_PLACEHOLDER}",
            f"--module={compiled_vmfb_path.resolve()}",
            "--hip_use_streams=true",
            "--hip_allow_inline_execution=true",
            "--batch_size=1000",
            "--benchmark_repetitions=3",
            "--benchmark_format=json",
        ]

        return command

    def get_model_compile_timeout_s(self) -> int:
        return 300

    def get_model_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        mlir_spec_path = candidate_tracker.spec_path
        assert mlir_spec_path is not None
        target_dir = mlir_spec_path.resolve().parent.parent.parent
        output_name = f"unet_candidate_{candidate_tracker.candidate_id}.vmfb"
        command = [
            "compile-punet-base.sh",
            "iree-compile",
            "gfx942",
            f"{mlir_spec_path.resolve()}",
            "./punet.mlir",
            "-o",
            (target_dir / output_name).as_posix(),
        ]
        return command

    def get_model_benchmark_timeout_s(self) -> int:
        return 180

    def get_model_benchmark_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        unet_candidate_path = candidate_tracker.compiled_model_path
        assert unet_candidate_path is not None

        command = [
            "iree-benchmark-module",
            f"--device={libtuner.DEVICE_ID_PLACEHOLDER}",
            "--hip_use_streams=true",
            "--hip_allow_inline_execution=true",
            "--device_allocator=caching",
            f"--module={unet_candidate_path.resolve()}",
            "--parameters=model=punet.irpa",
            "--function=main",
            "--input=1x4x128x128xf16",
            "--input=1xsi32",
            "--input=2x64x2048xf16",
            "--input=2x1280xf16",
            "--input=2x6xf16",
            "--input=1xf16",
            "--benchmark_repetitions=5",
            "--benchmark_format=json",
        ]
        return command

    def get_iree_compile_flags(self) -> list[str]:
        return []

    def get_iree_benchmark_module_flags(self) -> list[str]:
        return []

    def get_benchmark_timeout_s(self) -> int:
        return 0


def main():
    args = libtuner.parse_arguments()
    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[libtuner.CandidateTracker] = []
    punet_client = PunetClient()
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    print("Generating candidates...")
    candidates = libtuner.generate_candidates(args, path_config, candidate_trackers)
    print(f"Stored candidates in {path_config.candidates_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.generate_candidates:
        return

    print("Compiling candidates...")
    compiled_candidates = libtuner.compile_dispatches(
        args, path_config, candidates, candidate_trackers, punet_client
    )
    print(f"Compiled files are stored in {path_config.compiled_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.compile_dispatches:
        return

    print("Benchmarking compiled candidates...")
    top_candidates = libtuner.benchmark_dispatches(
        args, path_config, compiled_candidates, candidate_trackers, punet_client
    )
    print(f"Stored results in {path_config.output_unilog}\n")
    if stop_after_phase == libtuner.ExecutionPhases.benchmark_dispatches:
        return

    print(f"Compiling top model candidates...")
    punet_candidates = libtuner.compile_models(
        args, path_config, top_candidates, candidate_trackers, punet_client
    )
    print(f"Model candidates compiled in {path_config.base_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.compile_models:
        return

    print("Benchmarking model candidates...")
    libtuner.benchmark_models(
        args, path_config, punet_candidates, candidate_trackers, punet_client
    )
    print(f"Stored results in {path_config.output_unilog}")
    if stop_after_phase == libtuner.ExecutionPhases.benchmark_models:
        return

    libtuner.summerize_top_candidates(path_config, candidate_trackers)
    print(f"Stored top candidates info in {path_config.result_summary_log}\n")

    libtuner.save_pickle(path_config.candidate_trackers_pkl, candidate_trackers)
    print(f"Candidate trackers are saved in {path_config.candidate_trackers_pkl}\n")

    print("Check the detailed execution logs in:")
    print(path_config.run_log)

    for candidate in candidate_trackers:
        libtuner.logging.debug(candidate)
        if args.verbose:
            print(candidate)
