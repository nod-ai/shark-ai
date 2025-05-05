# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import shutil
from pathlib import Path
from sharktuner import libtuner
from sharktuner.common import *


class ModelTuner(libtuner.TuningClient):
    def __init__(self, tuner_context: libtuner.TunerContext):
        super().__init__(tuner_context)
        self.compile_flags: list[str] = []
        self.benchmark_flags: list[str] = []
        self.compile_timeout: int = 16
        self.benchmark_timeout: int = 16

    def get_iree_compile_flags(self) -> list[str]:
        return self.compile_flags

    def get_iree_compile_timeout_s(self) -> int:
        return self.compile_timeout

    def get_iree_benchmark_module_flags(self) -> list[str]:
        return self.benchmark_flags

    def get_benchmark_timeout_s(self) -> int:
        return self.benchmark_timeout


def read_flags_file(flags_file: str) -> list[str]:
    if not flags_file:
        return []

    with open(flags_file) as file:
        return file.read().splitlines()


def arg_parse() -> argparse.Namespace:
    # Custom arguments for the example tuner file.
    parser = argparse.ArgumentParser(description="Autotune sample script")
    client_args = parser.add_argument_group("Shark Tuner Options")
    client_args.add_argument(
        "model_file", type=Path, help="Path to the model file to tune (.mlir)"
    )
    client_args.add_argument(
        "--model-tuner-num-dispatch-candidates",
        type=int,
        default=None,
        help="Number of dispatch candidates to keep for model benchmarks.",
    )
    client_args.add_argument(
        "--model-tuner-num-model-candidates",
        type=int,
        default=None,
        help="Number of model candidates to produce after tuning.",
    )
    client_args.add_argument(
        "--compile-flags-file",
        type=str,
        default="",
        help="Path to the flags file for iree-compile.",
    )
    client_args.add_argument(
        "--model-benchmark-flags-file",
        type=str,
        default="",
        help="Path to the flags file for iree-benchmark-module for model benchmarking.",
    )
    client_args.add_argument(
        "--output-td-spec",
        type=Path,
        help="Path to write the best tuned spec. Dumps the best tuned model spec by default, and the best tuned dispatch spec when --stop-after is set to 'benchmark-dispatches'.",
        default="tuning-spec.mlir",
    )
    # Remaining arguments come from libtuner
    args = libtuner.parse_arguments(parser)
    return args


def main() -> None:
    args = arg_parse()

    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    # TODO(Max191): Make candidate_trackers internal to TuningClient.
    candidate_trackers: list[libtuner.CandidateTracker] = []
    stop_after_phase: str = args.stop_after

    print("[WARNING] SHARK Tuner is still experimental")
    root_logger = libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    compile_flags: list[str] = read_flags_file(args.compile_flags_file)
    model_benchmark_flags: list[str] = read_flags_file(args.model_benchmark_flags_file)

    summary_log_file = path_config.base_dir / "summary.log"
    summary_handler = logging.FileHandler(summary_log_file)
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    print("Generating candidate tuning specs...")
    with TunerContext(logger=root_logger) as tuner_context:
        tuner_context.logger.addHandler(summary_handler)
        model_tuner = ModelTuner(tuner_context)
        candidates = libtuner.generate_candidate_specs(
            args, path_config, candidate_trackers, model_tuner
        )
        print(f"Stored candidate tuning specs in {path_config.specs_dir}\n")
        if stop_after_phase == libtuner.ExecutionPhases.generate_candidates:
            return

        print("Compiling dispatch candidates...")
        model_tuner.compile_flags = compile_flags + [
            "--compile-from=executable-sources"
        ]
        compiled_candidates = libtuner.compile(
            args, path_config, candidates, candidate_trackers, model_tuner
        )
        if stop_after_phase == libtuner.ExecutionPhases.compile_dispatches:
            return

        message = "Benchmarking compiled dispatch candidates..."
        print(message)
        logging.info(message)
        model_tuner.benchmark_flags = ["--input=1", "--benchmark_repetitions=3"]
        top_candidates = libtuner.benchmark(
            args,
            compiled_candidates,
            candidate_trackers,
            model_tuner,
            args.model_tuner_num_dispatch_candidates,
        )
        logging.info(f"Top dispatch candidates: {top_candidates}")
        for id in top_candidates:
            logging.info(f"{candidate_trackers[id].spec_path.resolve()}")
        if stop_after_phase == libtuner.ExecutionPhases.benchmark_dispatches:
            top_spec_path = (
                path_config.specs_dir
                / path_config.get_candidate_spec_filename(top_candidates[0])
            )
            shutil.copy(top_spec_path, args.output_td_spec)
            print(f"Saved top spec ({top_spec_path}) to {args.output_td_spec}")
            return

        print("Compiling models with top candidates...")
        model_tuner.compile_flags = compile_flags
        model_tuner.compile_timeout = 120
        compiled_model_candidates = libtuner.compile(
            args,
            path_config,
            top_candidates,
            candidate_trackers,
            model_tuner,
            args.model_file,
        )
        if stop_after_phase == libtuner.ExecutionPhases.compile_models:
            return

        message = "Benchmarking compiled model candidates..."
        print(message)
        logging.info(message)
        model_tuner.benchmark_flags = model_benchmark_flags
        model_tuner.benchmark_timeout = 60
        top_model_candidates = libtuner.benchmark(
            args,
            compiled_model_candidates,
            candidate_trackers,
            model_tuner,
            args.model_tuner_num_model_candidates,
        )
        logging.info(f"Top model candidates: {top_model_candidates}")
        for id in top_model_candidates:
            logging.info(f"{candidate_trackers[id].spec_path.resolve()}")
        print(f"Top model candidates: {top_model_candidates}")

        top_spec_path = path_config.specs_dir / path_config.get_candidate_spec_filename(
            top_model_candidates[0]
        )
        shutil.copy(top_spec_path, args.output_td_spec)
        print(f"Saved top spec ({top_spec_path}) to {args.output_td_spec}")

        print("Check the detailed execution logs in:")
        print(path_config.run_log.resolve())
        print("Check the summary in:")
        print(summary_log_file.resolve())
