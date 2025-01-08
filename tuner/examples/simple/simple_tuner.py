# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path
from tuner import libtuner


class TestTuner(libtuner.TuningClient):
    def __init__(self):
        super().__init__()
        self.compile_flags = ["--compile-from=executable-sources"]
        self.benchmark_flags = ["--benchmark_repetitions=3", "--input=1"]

    def get_iree_compile_flags(self) -> list[str]:
        return self.compile_flags

    def get_iree_benchmark_module_flags(self) -> list[str]:
        return self.benchmark_flags

    def get_benchmark_timeout_s(self) -> int:
        return 10

    # TODO(Max191): Remove the following unused abstract functions once they
    # are removed from the TuningClient definition.
    def get_dispatch_benchmark_timeout_s(self) -> int:
        return 0

    def get_dispatch_compile_timeout_s(self) -> int:
        return 0

    def get_dispatch_compile_command(
        self, candidate_tracker: libtuner.CandidateTracker
    ) -> list[str]:
        return []

    def get_dispatch_benchmark_command(
        self,
        candidate_tracker: libtuner.CandidateTracker,
    ) -> list[str]:
        return []

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


def main():
    # Custom arguments for the test file.
    parser = argparse.ArgumentParser(description="Autotune test script")
    test_args = parser.add_argument_group("Example Test Options")
    test_args.add_argument(
        "simple_model_file", type=Path, help="Path to the model file to tune (.mlir)"
    )
    test_args.add_argument(
        "--simple-num-dispatch-candidates",
        type=int,
        default=None,
        help="Number of dispatch candidates to keep for model benchmarks.",
    )
    test_args.add_argument(
        "--simple-num-model-candidates",
        type=int,
        default=None,
        help="Number of model candidates to produce after tuning.",
    )
    test_args.add_argument(
        "--simple-hip-target",
        type=str,
        default="gfx942",
        help="Hip target for tuning.",
    )
    # Remaining arguments come from libtuner
    args = libtuner.parse_arguments(parser)

    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    # TODO(Max191): Make candidate_trackers internal to TuningClient.
    candidate_trackers: list[libtuner.CandidateTracker] = []
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    print("Generating candidate tuning specs...")
    test_tuner = TestTuner()
    candidates = libtuner.generate_candidate_specs(
        args, path_config, candidate_trackers, test_tuner
    )
    print(f"Stored candidate tuning specs in {path_config.specs_dir}\n")
    if stop_after_phase == libtuner.ExecutionPhases.generate_candidates:
        return

    print("Compiling dispatch candidates...")
    compiled_candidates = libtuner.compile(
        args, path_config, candidates, candidate_trackers, test_tuner
    )
    if stop_after_phase == libtuner.ExecutionPhases.compile_dispatches:
        return

    print("Benchmarking compiled dispatch candidates...")
    top_candidates = libtuner.benchmark(
        args,
        path_config,
        compiled_candidates,
        candidate_trackers,
        test_tuner,
        args.simple_num_dispatch_candidates,
    )
    if stop_after_phase == libtuner.ExecutionPhases.benchmark_dispatches:
        return

    print("Compiling models with top candidates...")
    test_tuner.compile_flags = [
        "--iree-hal-target-backends=rocm",
        f"--iree-hip-target={args.simple_hip_target}",
    ]
    compiled_model_candidates = libtuner.compile(
        args,
        path_config,
        top_candidates,
        candidate_trackers,
        test_tuner,
        args.simple_model_file,
    )
    if stop_after_phase == libtuner.ExecutionPhases.compile_models:
        return

    print("Benchmarking compiled model candidates...")
    test_tuner.benchmark_flags = [
        "--benchmark_repetitions=3",
        "--input=2048x2048xf16",
        "--input=2048x2048xf16",
    ]
    top_model_candidates = libtuner.benchmark(
        args,
        path_config,
        compiled_model_candidates,
        candidate_trackers,
        test_tuner,
        args.simple_num_model_candidates,
    )

    print(f"Top model candidates: {top_model_candidates}")

    print("Check the detailed execution logs in:")
    print(path_config.run_log.resolve())

    for candidate in candidate_trackers:
        libtuner.logging.debug(candidate)
