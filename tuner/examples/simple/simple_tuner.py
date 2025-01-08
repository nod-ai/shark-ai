# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path
from tuner import libtuner
from tuner.common import *


class SimpleTuner(libtuner.TuningClient):
    def __init__(self, tuner_context: libtuner.TunerContext):
        super().__init__(tuner_context)
        self.compile_flags = ["--compile-from=executable-sources"]
        self.benchmark_flags = ["--benchmark_repetitions=3", "--input=1"]

    def get_iree_compile_flags(self) -> list[str]:
        return self.compile_flags

    def get_iree_benchmark_module_flags(self) -> list[str]:
        return self.benchmark_flags

    def get_benchmark_timeout_s(self) -> int:
        return 10


def main():
    # Custom arguments for the example tuner file.
    parser = argparse.ArgumentParser(description="Autotune sample script")
    client_args = parser.add_argument_group("Simple Example Tuner Options")
    client_args.add_argument(
        "simple_model_file", type=Path, help="Path to the model file to tune (.mlir)"
    )
    client_args.add_argument(
        "--simple-num-dispatch-candidates",
        type=int,
        default=None,
        help="Number of dispatch candidates to keep for model benchmarks.",
    )
    client_args.add_argument(
        "--simple-num-model-candidates",
        type=int,
        default=None,
        help="Number of model candidates to produce after tuning.",
    )
    client_args.add_argument(
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
    with TunerContext() as tuner_context:
        simple_tuner = SimpleTuner(tuner_context)
        candidates = libtuner.generate_candidate_specs(
            args, path_config, candidate_trackers, simple_tuner
        )
        print(f"Stored candidate tuning specs in {path_config.specs_dir}\n")
        if stop_after_phase == libtuner.ExecutionPhases.generate_candidates:
            return

        print("Compiling dispatch candidates...")
        compiled_candidates = libtuner.compile(
            args, path_config, candidates, candidate_trackers, simple_tuner
        )
        if stop_after_phase == libtuner.ExecutionPhases.compile_dispatches:
            return

        print("Benchmarking compiled dispatch candidates...")
        top_candidates = libtuner.benchmark(
            args,
            path_config,
            compiled_candidates,
            candidate_trackers,
            simple_tuner,
            args.simple_num_dispatch_candidates,
        )
        if stop_after_phase == libtuner.ExecutionPhases.benchmark_dispatches:
            return

        print("Compiling models with top candidates...")
        simple_tuner.compile_flags = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={args.simple_hip_target}",
        ]
        compiled_model_candidates = libtuner.compile(
            args,
            path_config,
            top_candidates,
            candidate_trackers,
            simple_tuner,
            args.simple_model_file,
        )
        if stop_after_phase == libtuner.ExecutionPhases.compile_models:
            return

        print("Benchmarking compiled model candidates...")
        simple_tuner.benchmark_flags = [
            "--benchmark_repetitions=3",
            "--input=2048x2048xf16",
            "--input=2048x2048xf16",
        ]
        top_model_candidates = libtuner.benchmark(
            args,
            path_config,
            compiled_model_candidates,
            candidate_trackers,
            simple_tuner,
            args.simple_num_model_candidates,
        )

        print(f"Top model candidates: {top_model_candidates}")

        print("Check the detailed execution logs in:")
        print(path_config.run_log.resolve())

    for candidate in candidate_trackers:
        libtuner.logging.debug(candidate)
