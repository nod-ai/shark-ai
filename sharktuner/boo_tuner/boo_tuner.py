# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Optional
from typing_extensions import override

from sharktuner import common, libtuner


class BooTuner(libtuner.TuningClient):
    """Tuning client for BOO (Bag of Ops) kernels."""

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
        # BooTuner has only one phase, so prune candidates if all are slower than baseline.
        return True


def arg_parse() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    boo_args = parser.add_argument_group("BOO Tuner Options")
    boo_args.add_argument(
        "--commands-file",
        type=str,
        help="Read MIOpen commands from a file (one per line).",
    )
    boo_args.add_argument(
        "--output-td-spec",
        type=Path,
        default="tuning-spec.mlir",
        help="Path to write the best tuned spec.",
    )
    boo_args.add_argument(
        "--tmp-dir", type=str, default="", help="Directory to save temporary files."
    )
    boo_args.add_argument(
        "--boo-tuner-num-dispatch-candidates",
        type=int,
        default=None,
        help="Number of dispatch candidates to keep for benchmarking.",
    )
    boo_args.add_argument(
        "--boo-dispatch-benchmark-timeout-mins",
        type=float,
        default=None,
        help="Timeout in minutes for dispatch benchmarking.",
    )

    # Insert placeholder input_file for libtuner (BOO generates files internally).
    sys.argv = [sys.argv[0], "boo.mlir"] + sys.argv[1:]
    args = libtuner.parse_arguments(parser, allow_unknown=True)

    if "--codegen-pipeline" not in sys.argv:
        # Default to tile_and_fuse for BOO operations.
        args.codegen_pipeline = libtuner.CodegenPipelines.llvmgpu_tile_and_fuse

    # Extract MIOpen operation arguments (parser now knows all BOO + libtuner arguments).
    _, miopen_op_args = parser.parse_known_args()

    return args, miopen_op_args


def main() -> None:
    args, miopen_op_args = arg_parse()
    path_config = libtuner.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)

    print("[WARNING] BOO Tuner is still experimental")
    root_logger = libtuner.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    if not args.dry_run:
        print("Validating devices")
        libtuner.validate_devices(args.devices)
        print("Validation successful!\n")

    # These imports are slow due to a pytorch dependency. Keeping them local.
    # helps get fast '--help' output.
    from iree.turbine.kernel.boo import runtime as boo_runtime
    from iree.turbine.kernel.boo.driver.launch import get_launchable
    from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry

    logging.getLogger("turbine").setLevel(logging.WARNING)

    # Split tab-separated arguments (for easier copy-pasting from TSV files).
    miopen_op_args = [a for arg in miopen_op_args for a in arg.split("\t")]
    # Load MIOpen commands from file if specified, otherwise use command-line arguments.
    mio_args: list[list[str]] = [miopen_op_args]
    commands_file: str | None = args.commands_file
    if commands_file:
        splitter: Callable[[str], list[str]] = lambda s: (
            s.strip().split("\t") if commands_file.endswith(".tsv") else shlex.split(s)
        )
        with open(commands_file) as f:
            mio_args = [
                splitter(s) + miopen_op_args
                for s in f.readlines()
                if s.strip() and not s.startswith("#")
            ]

    starter_td_spec: Path | None = args.starter_td_spec
    for idx, cli_args in enumerate(mio_args):
        message = f">>> ({idx+1}/{len(mio_args)}) {shlex.join(cli_args)}"
        print(message)
        logging.info(message)
        sig = BooOpRegistry.parse_command(cli_args, ignore_unhandled_args=True)
        if sig is None:
            raise ValueError(
                f"Boo op registry failed to parse '{shlex.join(cli_args)}'."
            )
        if args.tmp_dir:
            tmp_dir = Path(args.tmp_dir)
            # Make sure directory is empty.
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.mkdir(tmp_dir)
        else:
            tmp_dir = Path(tempfile.mkdtemp(dir="boo_tuner", prefix="boo-tuner-"))
        boo_cache_dir = tmp_dir / "boo_cache"

        # Run BOO compilation and extract source IR.
        with boo_runtime.use_cache_dir(boo_cache_dir):
            # Note: device="cuda" is correct for AMD GPUs.
            get_launchable(sig)(*sig.get_sample_args(device="cuda", seed=123))
        [op_cache_dir] = os.listdir(boo_cache_dir)
        op_cache_path = boo_cache_dir / op_cache_dir

        # Find the source MLIR file.
        [source_mlir_file] = [
            f for f in os.listdir(op_cache_path) if f.endswith(".mlir")
        ]
        source_mlir_path = op_cache_path / source_mlir_file
        print(f"source_mlir_path: {source_mlir_path}")

        # Find the compile command file.
        [compile_command_file] = [
            f for f in os.listdir(op_cache_path) if f.startswith("compile_command")
        ]
        with open(op_cache_path / compile_command_file) as f:
            compile_command = f.read().strip()

        # Use all turbine compile flags except the output file (-o *.vmfb).
        turbine_compile_flags = shlex.split(compile_command)

        # removing '-o' and the output .vmfb path.
        compile_args = ["iree-compile"]
        args_iter = iter(turbine_compile_flags[1:])
        for arg in args_iter:
            if arg == "-o":
                next(args_iter, None)
            else:
                compile_args.append(arg)

        # Add tuner-specific flags.
        benchmarks_dir = tmp_dir / "benchmarks"
        compile_args.extend(
            [
                "--iree-config-add-tuner-attributes",
                "--iree-hal-dump-executable-benchmarks-to",
                str(benchmarks_dir),
                "-o",
                os.devnull,
            ]
        )
        logging.info(f"> {shlex.join(compile_args)}")
        subprocess.run(compile_args)

        summary_log_file = path_config.base_dir / "summary.log"
        summary_handler = logging.FileHandler(summary_log_file)
        summary_handler.setLevel(logging.INFO)
        summary_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        should_cleanup_tmp_dir = not args.tmp_dir

        # Process all generated benchmark files.
        benchmark_files = list(os.listdir(benchmarks_dir))
        for benchmark_file in benchmark_files:
            benchmark_path = benchmarks_dir / benchmark_file
            logging.info(f"Tuning benchmark: {benchmark_path}")

            try:
                args.input_file = benchmark_path
                # Only use starter spec if it exists and the file is present.
                if starter_td_spec and starter_td_spec.exists():
                    args.starter_td_spec = starter_td_spec
                else:
                    args.starter_td_spec = None

                print("Generating candidate tuning specs...")
                with common.TunerContext(logger=root_logger) as tuner_context:
                    tuner_context.logger.addHandler(summary_handler)
                    boo_tuner = BooTuner(tuner_context)
                    candidates = libtuner.generate_candidate_specs(
                        args, path_config, boo_tuner
                    )
                    print(f"Stored candidate tuning specs in {path_config.specs_dir}\n")

                    print("Compiling dispatch candidates...")
                    boo_tuner.compile_flags = ["--compile-from=executable-sources"]
                    compiled_candidates = libtuner.compile(
                        args, path_config, candidates, boo_tuner
                    )

                    message = "Benchmarking compiled dispatch candidates..."
                    print(message)
                    logging.info(message)
                    boo_tuner.benchmark_flags = [
                        "--input=1",
                        "--benchmark_repetitions=3",
                    ]
                    top_candidates = libtuner.benchmark(
                        args,
                        compiled_candidates,
                        boo_tuner,
                        args.boo_tuner_num_dispatch_candidates,
                        args.boo_dispatch_benchmark_timeout_mins,
                    )

                    if not top_candidates:
                        logging.critical(
                            "No tuning candidates performed better than the baseline."
                        )
                    else:
                        logging.info(f"Top dispatch candidates: {top_candidates}")
                        for id in top_candidates:
                            logging.info(
                                f"{boo_tuner.candidate_trackers[id].spec_path.resolve()}"
                            )

                        # Save the best (first) tuning spec to output file.
                        best_candidate_id = top_candidates[0]
                        best_spec_path = boo_tuner.candidate_trackers[
                            best_candidate_id
                        ].spec_path
                        shutil.copy(best_spec_path, args.output_td_spec)
                        logging.info(
                            f"Saved best tuning spec to: {args.output_td_spec}"
                        )
                        print(f"Saved best tuning spec to: {args.output_td_spec}")

                        # Update starter spec for next benchmark iteration.
                        starter_td_spec = args.output_td_spec

            except Exception as err:
                traceback.print_exception(err)
                should_cleanup_tmp_dir = False

        if path_config.run_log is not None:
            print("\nCheck the detailed execution logs in:")
            print(path_config.run_log.resolve())
        print("Check the summary in:")
        print(summary_log_file.resolve())

        if should_cleanup_tmp_dir:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
