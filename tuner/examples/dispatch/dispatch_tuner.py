# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Sample Usage:

python -m examples.dispatch benchmark.mlir --lhs-dims=bmk --rhs-dims=bkn --tile-dims=*mnk --devices=hip://0,hip://1 --num-candidates=64


Recommended Trial Run:

python -m examples.dispatch benchmark.mlir --num-candidates=10


Dry Run Test (no gpu required):

python -m examples.dispatch benchmark.mlir --num-candidates=64 --dry-run

"""

from tuner import libtuner
from pathlib import Path, PurePath
import os


class DispatchTuner(libtuner.TuningClient):
    def get_dispatch_compile_timeout_s(self) -> int:
        return 10

    def get_dispatch_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        assert candidate_tracker.dispatch_mlir_path is not None
        mlir_path: Path = candidate_tracker.dispatch_mlir_path
        script_dir = Path(__file__).resolve().parent
        command = [
            (script_dir / "compile_dispatch.sh").as_posix(),
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
            "--batch_size=1000",
            "--benchmark_repetitions=3",
            "--benchmark_format=json",
        ]

        return command

    def get_model_compile_timeout_s(self) -> int:
        return 0

    def get_model_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return []

    def get_model_benchmark_timeout_s(self) -> int:
        return 0

    def get_model_benchmark_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return []

    def get_iree_compile_flags(self) -> list[str]:
        return []

    def get_iree_benchmark_module_flags(self) -> list[str]:
        return []

    def get_benchmark_timeout_s(self) -> int:
        return 0


def main():
    args = libtuner.parse_arguments()
    path_config = libtuner.PathConfig()
    # These will not be used, so always default to the empty config in the script dir.
    script_dir = Path(__file__).resolve().parent
    path_config.global_config_prolog_mlir = (
        script_dir / path_config.global_config_prolog_mlir
    )
    path_config.global_config_epilog_mlir = (
        script_dir / path_config.global_config_epilog_mlir
    )
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[libtuner.CandidateTracker] = []
    dispatch_tuner = DispatchTuner()
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
        args, path_config, candidates, candidate_trackers, dispatch_tuner
    )
    print(f"Compiled files are stored in {path_config.compiled_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.compile_dispatches:
        return

    print("Benchmarking compiled candidates...")
    top_candidates = libtuner.benchmark_dispatches(
        args, path_config, compiled_candidates, candidate_trackers, dispatch_tuner
    )
    print(f"\nStored results in {path_config.output_unilog.resolve()}\n")
    if stop_after_phase == libtuner.ExecutionPhases.benchmark_dispatches:
        return

    libtuner.save_pickle(path_config.candidate_trackers_pkl, candidate_trackers)
    print(f"Candidate trackers are saved in {path_config.candidate_trackers_pkl}\n")

    print("Check the detailed execution logs in:")
    print(path_config.run_log.resolve())

    for candidate in candidate_trackers:
        libtuner.logging.debug(candidate)
