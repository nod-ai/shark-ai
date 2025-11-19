# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import argparse
from pathlib import Path
from typing import Optional
from typing_extensions import override

from amdsharktuner import common, libtuner


class DispatchTuner(libtuner.TuningClient):
    def __init__(self, tuner_context: common.TunerContext):
        super().__init__(tuner_context)
        self.compile_flags: list[str] = []
        self.benchmark_flags: list[str] = []
        self.compile_timeout: Optional[float] = 16
        self.benchmark_timeout: Optional[float] = None
        self.auto_benchmark_timeout: bool = True

    @override
    def get_iree_compile_flags(self) -> list[str]:
        return self.compile_flags

    @override
    def get_iree_compile_timeout_s(self) -> Optional[float]:
        return self.compile_timeout

    @override
    def get_iree_benchmark_module_flags(self) -> list[str]:
        return self.benchmark_flags

    @override
    def get_iree_benchmark_timeout_s(self) -> Optional[float]:
        return self.benchmark_timeout

    @override
    def is_auto_iree_benchmark_timeout(self) -> bool:
        return self.auto_benchmark_timeout

    @override
    def should_prune_slower_candidates(self) -> bool:
        # DispatchTuner has only one phase, so prune candidates if all are slower than baseline.
        return True


def read_flags_file(flags_file: str) -> list[str]:
    if not flags_file:
        return []

    with open(flags_file) as file:
        return file.read().splitlines()


def arg_parse() -> argparse.Namespace:
    # Custom arguments for the example tuner file.
    parser = argparse.ArgumentParser(description="Autotune sample script")
    client_args = parser.add_argument_group("amdshark Tuner Options")
    client_args.add_argument(
        "dispatch_file", type=Path, help="Path to the dispatch file to tune (.mlir)"
    )
    client_args.add_argument(
        "--dispatch-tuner-num-dispatch-candidates",
        type=int,
        default=None,
        help="Number of dispatch candidates to keep for dispatch benchmarks.",
    )
    client_args.add_argument(
        "--compile-flags-file",
        type=str,
        default="",
        help="Path to the flags file for iree-compile.",
    )
    client_args.add_argument(
        "--output-td-spec",
        type=Path,
        help="Path to write the best tuned spec. Dumps the best tuned dispatch spec by default, and the best tuned dispatch spec when --stop-after is set to 'benchmark-dispatches'.",
        default="tuning-spec.mlir",
    )
    client_args.add_argument(
        "--dispatch-benchmark-timeout-mins",
        type=float,
        default=None,
        help="Time budget in minutes for dispatch benchmark phase.",
    )
    # Remaining arguments come from libtuner.
    args = libtuner.parse_arguments(parser)
    return args


def main() -> None:
    args = arg_parse()

    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)

    print("[WARNING] amdshark Tuner is still experimental")
    root_logger = libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    compile_flags: list[str] = read_flags_file(args.compile_flags_file)

    summary_log_file = path_config.base_dir / "summary.log"
    summary_handler = logging.FileHandler(summary_log_file)
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    print("Generating candidate tuning specs...")
    with common.TunerContext(logger=root_logger) as tuner_context:
        tuner_context.logger.addHandler(summary_handler)
        dispatch_tuner = DispatchTuner(tuner_context)
        candidates = libtuner.generate_candidate_specs(
            args, path_config, dispatch_tuner
        )
        print(f"Stored candidate tuning specs in {path_config.specs_dir}\n")

        print("Compiling dispatch candidates...")
        dispatch_tuner.compile_flags = compile_flags + [
            "--compile-from=executable-sources"
        ]

        compiled_candidates = libtuner.compile(
            args, path_config, candidates, dispatch_tuner
        )

        message = "Benchmarking compiled dispatch candidates..."
        print(message)
        logging.info(message)
        dispatch_tuner.benchmark_flags = ["--input=1", "--benchmark_repetitions=3"]
        top_candidates = libtuner.benchmark(
            args,
            compiled_candidates,
            dispatch_tuner,
            args.dispatch_tuner_num_dispatch_candidates,
            args.dispatch_benchmark_timeout_mins,
        )
        if not top_candidates:
            logging.critical("No tuning candidates performed better than the baseline.")
        else:
            logging.info(f"Top dispatch candidates: {top_candidates}")
            for id in top_candidates:
                logging.info(
                    f"{dispatch_tuner.candidate_trackers[id].spec_path.resolve()}"
                )

        if path_config.run_log is not None:
            print("Check the detailed execution logs in:")
            print(path_config.run_log.resolve())
        print("Check the summary in:")
        print(summary_log_file.resolve())

        output_csv_name = f"{args.dispatch_file.stem}_candidate_analysis.csv"
        csv_path = Path(path_config.base_dir) / output_csv_name

        libtuner.candidate_ordering.export_record_to_csv(
            dispatch_tuner.tuning_records, csv_path
        )
        print(f"Wrote tuning records CSV: {csv_path}")
