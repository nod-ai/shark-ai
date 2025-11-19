# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import contextlib
import gc
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing_extensions import override

from amdsharktuner import common, libtuner


class BooPathConfig(libtuner.PathConfig):
    """Path configuration for BOO tuner with BOO-specific directory naming."""

    def _name_base_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        base_dir = Path(f"./boo_tuning_{timestamp}")
        return base_dir

    def create_benchmark_path_config(self, benchmark_name: str) -> libtuner.PathConfig:
        """Create a PathConfig for a specific benchmark under the main BOO tuning directory."""
        base_dir = self.base_dir

        class BenchmarkPathConfig(libtuner.PathConfig):
            def _name_base_dir(self) -> Path:
                return base_dir / benchmark_name

        return BenchmarkPathConfig()


class BooTuner(libtuner.TuningClient):
    """Tuning client for IREE Turbine's BOO (Bag of Ops) kernels."""

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


def insert_placeholder_input_file(argv: list[str]) -> list[str]:
    """Insert a placeholder input file for libtuner compatibility."""
    return [argv[0], "boo.mlir"] + argv[1:]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
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

    # Insert a placeholder input_file for libtuner (BOO generates files internally).
    sys.argv = insert_placeholder_input_file(sys.argv)
    args = libtuner.parse_arguments(parser, allow_unknown=True)

    if "--codegen-pipeline" not in sys.argv:
        # Default to tile_and_fuse for BOO operations.
        args.codegen_pipeline = libtuner.CodegenPipelines.llvmgpu_tile_and_fuse

    # Extract MIOpen operation arguments (parser now knows all BOO + libtuner arguments).
    _, miopen_op_args = parser.parse_known_args()

    return args, miopen_op_args


def load_commands_from_file_or_args(
    commands_file: str | None, miopen_op_args: list[str]
) -> list[list[str]]:
    # Split tab-separated arguments (for easier copy-pasting from TSV files).
    miopen_op_args = [a for arg in miopen_op_args for a in arg.split("\t")]

    # Load MIOpen commands from file if specified, otherwise use command-line arguments.
    if not commands_file:
        return [miopen_op_args]

    # Validate that miopen_op_args is empty when using a commands file.
    if miopen_op_args:
        raise ValueError(
            "Cannot specify both --commands-file and MIOpen operation arguments."
        )

    splitter: Callable[[str], list[str]] = lambda s: (
        s.strip().split("\t") if commands_file.endswith(".tsv") else shlex.split(s)
    )
    with open(commands_file) as f:
        return [
            splitter(s) for s in f.readlines() if s.strip() and not s.startswith("#")
        ]


def build_compile_args(compile_command: str, benchmarks_dir: Path) -> list[str]:
    """Build iree-compile arguments from turbine compile command."""
    turbine_compile_flags = shlex.split(compile_command)

    # Start with iree-compile and filter out -o flag from turbine flags.
    compile_args = ["iree-compile"]
    args_iter = iter(turbine_compile_flags[1:])
    for arg in args_iter:
        if arg == "-o":
            next(args_iter, None)  # Skip the output file path.
            continue

        compile_args.append(arg)

    # Add tuner-specific flags.
    compile_args.extend(
        [
            "--iree-config-add-tuner-attributes",
            "--iree-hal-dump-executable-benchmarks-to",
            str(benchmarks_dir),
            "-o",
            os.devnull,
        ]
    )

    return compile_args


def tune_boo_dispatch(
    benchmark_path: Path,
    args: argparse.Namespace,
    path_config: libtuner.PathConfig,
    root_logger: logging.Logger,
    summary_handler: logging.Handler,
    starter_td_spec: Path | None,
) -> Path | None:
    """Tune a single BOO dispatch."""
    args.input_file = benchmark_path
    # Only use starter spec if it exists and the file is present.
    if starter_td_spec and starter_td_spec.exists():
        args.starter_td_spec = starter_td_spec
    else:
        args.starter_td_spec = None

    logging.info("Generating candidate tuning specs...")
    with common.TunerContext(logger=root_logger) as tuner_context:
        tuner_context.logger.addHandler(summary_handler)
        boo_tuner = BooTuner(tuner_context)
        candidates = libtuner.generate_candidate_specs(args, path_config, boo_tuner)
        logging.info(f"Stored candidate tuning specs in {path_config.specs_dir}")

        logging.info("Compiling dispatch candidates...")
        boo_tuner.compile_flags = ["--compile-from=executable-sources"]
        compiled_candidates = libtuner.compile(args, path_config, candidates, boo_tuner)

        logging.info("Benchmarking compiled dispatch candidates...")
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
            logging.critical("No tuning candidates performed better than the baseline.")
            return None

        logging.info(f"Top dispatch candidates: {top_candidates}")
        for id in top_candidates:
            logging.info(f"{boo_tuner.candidate_trackers[id].spec_path.resolve()}")

        # Save the best (first) tuning spec to the output file.
        best_candidate_id = top_candidates[0]
        best_spec_path = boo_tuner.candidate_trackers[best_candidate_id].spec_path
        shutil.copy(best_spec_path, args.output_td_spec)
        logging.info(f"Saved best tuning spec to: {args.output_td_spec}")
        print(f"Saved best tuning spec to: {args.output_td_spec}")

    return args.output_td_spec


def process_boo_command(
    cli_args: list[str],
    args: argparse.Namespace,
    boo_path_config: BooPathConfig,
    root_logger: logging.Logger,
    starter_td_spec: Path | None,
) -> Path | None:
    """Process a single BOO command through compilation and tuning."""
    # These imports are slow due to a pytorch dependency. Keeping them local helps
    # make '--help' fast.
    import torch  # type: ignore
    from iree.turbine.kernel.boo import runtime as boo_runtime
    from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry
    from iree.turbine.kernel.boo.runtime import LaunchableRuntimeCache
    from iree.turbine.runtime.device import get_device_from_torch

    sig = BooOpRegistry.parse_command(cli_args, ignore_unhandled_args=True)
    if sig is None:
        raise ValueError(f"Boo op registry failed to parse '{shlex.join(cli_args)}'.")

    # Set up temporary directory.
    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir)
        if tmp_dir.exists():
            logging.warning(f"Removing existing temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using user-specified temporary directory: {tmp_dir}")
    else:
        tmp_dir = Path(tempfile.mkdtemp(dir="boo_tuner", prefix="boo_turbine_cache_"))
        logging.info(f"Created temporary directory: {tmp_dir}")
    boo_cache_dir = tmp_dir / "boo_cache"

    # Run BOO compilation and extract source IR.
    with boo_runtime.use_cache_dir(boo_cache_dir):
        # Reset torch compilation cache to ensure we don't hit compilation limits.
        torch.compiler.reset()
        # Reset BOO compilation cache so repeated commands will still generate cache entries.
        LaunchableRuntimeCache.reset()
        # The "iree_boo" backend offloads to IREE in cases where we expect
        # performance to be better, and falls back to pytorch otherwise. We use
        # the experimental backend here instead, as we want to use IREE in all
        # cases.
        # Note: device="cuda" is correct for AMD GPUs.
        device = torch.device("cuda:0")
        sig.get_compiled_module(backend="iree_boo_experimental")(
            *sig.get_sample_args(device=device, seed=123)
        )
        # Reclaim cached allocations from IREE and pytorch so benchmarks have all
        # memory available.
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.memory.empty_cache()
        get_device_from_torch(device).hal_device.allocator.trim()

    [op_cache_dir] = os.listdir(boo_cache_dir)
    op_cache_path = boo_cache_dir / op_cache_dir

    # Find the source MLIR file.
    [source_mlir_file] = [f for f in os.listdir(op_cache_path) if f.endswith(".mlir")]
    source_mlir_path = op_cache_path / source_mlir_file
    logging.debug(f"source_mlir_path: {source_mlir_path}")

    # Find the compile command file.
    [compile_command_file] = [
        f for f in os.listdir(op_cache_path) if f.startswith("compile_command")
    ]
    with open(op_cache_path / compile_command_file) as f:
        compile_command = f.read().strip()

    # Build compile arguments with tuner-specific flags.
    benchmarks_dir = tmp_dir / "benchmarks"
    compile_args = build_compile_args(compile_command, benchmarks_dir)

    logging.info(f"> {shlex.join(compile_args)}")
    subprocess.run(compile_args)

    best_spec_path = None

    # Process all generated benchmark files.
    benchmark_files = list(os.listdir(benchmarks_dir))
    for benchmark_file in benchmark_files:
        benchmark_path = benchmarks_dir / benchmark_file
        logging.info(f"Tuning benchmark: {benchmark_path}")

        # Extract benchmark name from filename (remove _benchmark.mlir suffix).
        benchmark_name = benchmark_file.replace("_benchmark.mlir", "")

        # Create a dedicated PathConfig for this benchmark.
        benchmark_path_config = boo_path_config.create_benchmark_path_config(
            benchmark_name
        )
        benchmark_path_config.base_dir.mkdir(parents=True, exist_ok=True)

        # Create benchmark-specific summary log.
        summary_log_file = benchmark_path_config.base_dir / "summary.log"
        with contextlib.closing(
            logging.FileHandler(summary_log_file)
        ) as summary_handler:
            summary_handler.setLevel(logging.INFO)
            summary_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )

            try:
                result = tune_boo_dispatch(
                    benchmark_path,
                    args,
                    benchmark_path_config,
                    root_logger,
                    summary_handler,
                    starter_td_spec,
                )
                if result:
                    best_spec_path = result

                if benchmark_path_config.run_log is not None:
                    print(f"\nCheck the detailed execution logs in:")
                    print(benchmark_path_config.run_log.resolve())
                print(f"Check the summary in:")
                print(summary_log_file.resolve())

            except Exception as err:
                traceback.print_exception(err)

    return args.output_td_spec if best_spec_path else None


def main() -> None:
    # Set saner defaults for pytorch/miopen environment variables. This affects
    # pytorch's inferred tensor layouts on AMDGPU, even when not actually using
    # MIOpen kernels, and are required for performance.
    os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")

    parsed_args: tuple[argparse.Namespace, list[str]] = parse_args()
    args, miopen_op_args = parsed_args

    assert not (
        args.commands_file and miopen_op_args
    ), "Cannot specify both --commands-file and MIOpen operation arguments"

    # Create main tuning directory.
    boo_path_config: BooPathConfig = BooPathConfig()
    boo_path_config.base_dir.mkdir(parents=True, exist_ok=True)

    root_logger = libtuner.setup_logging(args, boo_path_config)
    print(boo_path_config.run_log)

    logging.warning("BOO Tuner is still experimental")

    if not args.dry_run:
        logging.info("Validating devices")
        libtuner.validate_devices(args.devices)
        logging.info("Validation successful!")

    logging.getLogger("turbine").setLevel(logging.WARNING)

    mio_args = load_commands_from_file_or_args(args.commands_file, miopen_op_args)

    starter_td_spec: Path | None = args.starter_td_spec
    for idx, cli_args in enumerate(mio_args):
        message = f">>> ({idx+1}/{len(mio_args)}) {shlex.join(cli_args)}"
        logging.info(message)

        result_spec = process_boo_command(
            cli_args,
            args,
            boo_path_config,
            root_logger,
            starter_td_spec,
        )

        # Update starter spec for next iteration if tuning succeeded.
        if result_spec:
            starter_td_spec = result_spec


if __name__ == "__main__":
    main()
