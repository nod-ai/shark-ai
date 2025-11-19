# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Provides fundamental functions for tuning:
    - generate_candidate_specs()
    - compile()
    - benchmark()

Requires a wrapper Python script to import `libtuner`,
use the `TuningClient` API, customize compilation and benchmarking commands,
and implement a complete tuning loop for a specific model.
"""


import math
import signal
import sys
import shutil
import logging
import argparse
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import multiprocessing
import queue
from tqdm import tqdm
import hashlib
from dataclasses import dataclass, field
from typing import Type, Optional, Callable, Iterable, Any
from abc import ABC, abstractmethod
import subprocess
import tempfile
import os
import random
import time

import iree.runtime as ireert  # type: ignore
import iree.compiler as ireec  # type: ignore
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu, iree_codegen  # type: ignore
from . import (
    candidate_gen,
    common,
    dispatch_constraints,
    dispatch_parser,
    candidate_ordering,
)


# Default random seed.
DEFAULT_SHUFFLE_SEED = 42

# Default multiplier applied to the base running time to calculate a smart timeout.
# Example: timeout = base_running_time * DEFAULT_TIMEOUT_MUL.
DEFAULT_TIMEOUT_MUL = 1.2

# Default values for num_candidates and devices, change it as needed.
DEFAULT_NUM_CANDIDATES = 2048
DEFAULT_DEVICE_LIST = ["hip://0"]

# Default values for max number of workers.
DEFAULT_MAX_CPU_WORKERS = (
    multiprocessing.cpu_count() // 2
)  # the actual amount of worker that will be generated = min(max_cpu_workers, len(task_list)).

# Declare global variables at the module level for multiprocessing.
worker_id = None
device_id = None

# Declare special symbols for libtuner to search and locate.
DEVICE_ID_PLACEHOLDER = "!DEVICE_ID!"


@dataclass
class CandidateTracker:
    candidate_id: int
    mlir_path: Optional[Path] = None
    compiled_vmfb_path: Optional[Path] = None
    spec_path: Optional[Path] = None
    td_spec_str: Optional[str] = None
    knob_assignment: Optional[common.KnobAssignment] = None


@dataclass()
class PathConfig:
    # Dynamic paths.
    base_dir: Path = field(init=False)
    template_mlir: Path = field(init=False)
    candidates_dir: Path = field(init=False)
    compiled_dir: Path = field(init=False)
    specs_dir: Path = field(init=False)

    # To be set outside of class.
    run_log: Optional[Path] = field(init=False, default=None)

    def __post_init__(self):
        object.__setattr__(self, "base_dir", self._name_base_dir())
        object.__setattr__(self, "template_mlir", self.base_dir / "template.mlir")
        object.__setattr__(self, "candidates_dir", self.base_dir / "candidates")
        object.__setattr__(self, "compiled_dir", self.candidates_dir / "compiled")
        object.__setattr__(self, "specs_dir", self.candidates_dir / "specs")

    def _name_base_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        base_dir = Path(f"./tuning_{timestamp}")
        return base_dir

    def set_run_log(self, run_log: Path):
        object.__setattr__(self, "run_log", run_log)

    def get_candidate_spec_filename(self, candidate_id: int) -> str:
        return f"{candidate_id}_spec.mlir"

    def get_candidate_vmfb_filename(self, candidate_id: int) -> str:
        return f"{candidate_id}.vmfb"


class TuningClient(ABC):
    def __init__(self, tuner_context: common.TunerContext):
        self.tuner_context = tuner_context
        self.candidate_trackers: list[CandidateTracker] = []
        self.target_info: Optional[iree_gpu.TargetInfo] = None
        self.tuning_records: list[candidate_ordering.TuningRecord] = []

    @abstractmethod
    def get_iree_compile_flags(self) -> list[str]:
        pass

    @abstractmethod
    def get_iree_compile_timeout_s(self) -> Optional[float]:
        pass

    @abstractmethod
    def get_iree_benchmark_module_flags(self) -> list[str]:
        pass

    @abstractmethod
    def get_iree_benchmark_timeout_s(self) -> Optional[float]:
        """
        Returns benchmark timeout in seconds.
        If None, no timeout is applied.

        When auto timeout is enabled, this is used only for the baseline run.
        Otherwise, it applies to both baseline and candidate runs.
        """
        pass

    @abstractmethod
    def is_auto_iree_benchmark_timeout(self) -> bool:
        """
        Return True if tuner should automatically derive candidate benchmark timeouts
        based on the measured runtime of the baseline subprocess.
        """
        pass

    @abstractmethod
    def should_prune_slower_candidates(self) -> bool:
        """
        Determines function benchmark behavior when all candidates are slower than baseline.
        Return False if slower candidates could improve in later phases (e.g., different
        fusions in full model), function benchmark returns top N candidates. Return True for
        final evaluation, function benchmark returns empty list.
        """
        pass


@dataclass
class CompilePack:
    iree_compile_flags: list[str]
    iree_compile_timeout: Optional[float]
    candidate_tracker: CandidateTracker


@dataclass
class BenchmarkPack:
    iree_benchmark_module_flags: list[str]
    benchmark_timeout: Optional[float]
    candidate_tracker: CandidateTracker


@dataclass
class BenchmarkResult:
    candidate_id: int
    time: float
    device_id: str

    def is_valid(self) -> bool:
        return math.isfinite(self.time)

    def __iter__(self):
        return iter((self.candidate_id, self.time, self.device_id))


def unit_to_microseconds(real_time: float, time_unit: str) -> float:
    unit_conversions = {
        "s": 1e6,
        "ms": 1e3,
        "us": 1,
        "ns": 1e-3,
    }

    assert time_unit in unit_conversions, f"Unsupported time unit: {time_unit}"

    return real_time * unit_conversions[time_unit]


def extract_driver_names(user_devices: list[str]) -> set[str]:
    """Extract driver names from the user devices"""
    return {device.split("://")[0] for device in user_devices}


def fetch_available_devices(drivers: list[str]) -> list[str]:
    """
    Extract all available devices on the user's machine for the provided drivers.
    Only the user provided drivers will be queried.
    """
    all_device_ids: list[str] = []

    for driver_name in drivers:
        try:
            driver = ireert.get_driver(driver_name)
            devices = driver.query_available_devices()
            all_device_ids.extend(
                f"{driver_name}://{device['path']}" for device in devices
            )
            all_device_ids.extend(
                f"{driver_name}://{device['device_id'] - 1}" for device in devices
            )
        except ValueError as e:
            handle_error(
                condition=True,
                msg=f"Could not initialize driver {driver_name}: {e}",
                error_type=ValueError,
                exit_program=True,
            )

    return all_device_ids


def parse_devices(devices_str: str) -> list[str]:
    """
    Parse a comma-separated list of device IDs e.g.:
    --devices=hip://0,local-sync://default -> ["hip://0", "local-sync://default"]).
    """
    devices = [device.strip() for device in devices_str.split(",")]
    for device in devices:
        if "://" not in device or not device:
            handle_error(
                condition=True,
                msg=f"Invalid device list: {devices_str}. Error: {ValueError()}",
                error_type=argparse.ArgumentTypeError,
            )
    return devices


def validate_devices(user_devices: list[str]) -> None:
    """Validates the user provided devices against the devices extracted by the IREE Runtime"""
    user_drivers = extract_driver_names(user_devices)

    available_devices = fetch_available_devices(list(user_drivers))

    for device in user_devices:
        handle_error(
            condition=(device not in available_devices),
            msg=f"Invalid device specified: {device}\nFetched available devices: {available_devices}",
            error_type=argparse.ArgumentError,
            exit_program=True,
        )


class ExecutionPhases(str, Enum):
    dont_stop = ""
    generate_candidates = "generate-candidates"
    compile_dispatches = "compile-dispatches"
    benchmark_dispatches = "benchmark-dispatches"
    compile_models = "compile-models"
    benchmark_models = "benchmark-models"


class CodegenPipelines(str, Enum):
    llvmgpu_vector_distribute = "llvmgpu_vector_distribute"
    llvmgpu_tile_and_fuse = "llvmgpu_tile_and_fuse"


def parse_arguments(
    initial_parser: Optional[argparse.ArgumentParser] = None,
    allow_unknown: bool = False,
) -> argparse.Namespace:
    parser = initial_parser
    if parser is None:
        parser = argparse.ArgumentParser(description="Autotune script")

    # Required arguments.
    required_args = parser.add_argument_group("Required Options")
    required_args.add_argument(
        "input_file", type=Path, help="Path to the input benchmark file (.mlir)"
    )

    # General options.
    general_args = parser.add_argument_group("General Options")
    # Note: No -v shorthand to avoid conflicts with MIOpen driver arguments (e.g., -v for vertical stride/dilation).
    general_args.add_argument(
        "--verbose", action="store_true", help="Enable verbose output to stdout"
    )
    general_args.add_argument(
        "--devices",
        type=parse_devices,
        default=DEFAULT_DEVICE_LIST,
        help="Comma-separated list of device IDs (e.g., --devices=hip://,hip://GPU-UUID).",
    )
    general_args.add_argument(
        "--max-cpu-workers",
        type=int,
        default=DEFAULT_MAX_CPU_WORKERS,
        help=f"Max number of workers for CPU-bounding tasks (default: {DEFAULT_MAX_CPU_WORKERS}, the number of CPUs in current system)",
    )
    general_args.add_argument(
        "--stop-after",
        choices=[x.value for x in ExecutionPhases],
        default=ExecutionPhases.dont_stop,
        help="Stop execution after specified phase",
    )
    general_args.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not attempt to run any modules or initialize the IREE runtime",
    )

    # candidate_gen.tune() options.
    candidate_gen_args = parser.add_argument_group("Candidate Generation Options")
    candidate_gen_args.add_argument(
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Number of candidates to be generated by candidate_gen.py (default: {DEFAULT_NUM_CANDIDATES})",
    )
    general_args.add_argument(
        "--candidate-order",
        choices=[s.value for s in candidate_ordering.CandidateOrderKind],
        default=candidate_ordering.CandidateOrderKind.shuffle.value,
        help="How to order generated candidates for compilation and benchmarking.",
    )
    candidate_gen_args.add_argument(
        "--search-space-shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed for shuffling candidates during generation (default: {DEFAULT_SHUFFLE_SEED}).",
    )
    candidate_gen_args.add_argument(
        "--enable-random-seed",
        action="store_true",
        help="Uses a non-deterministic random shuffle seed for candidate generation phase.",
    )
    candidate_gen_args.add_argument(
        "--num-subgroups",
        help="Number of subgroups per workgroup to use. (-1 == unconstrained)",
        type=int,
        default=-1,
    )
    candidate_gen_args.add_argument(
        "--lhs-dims", help="Map of LHS matmul dims", type=str, default="mk"
    )
    candidate_gen_args.add_argument(
        "--rhs-dims", help="Map of RHS matmul dims", type=str, default="nk"
    )
    candidate_gen_args.add_argument(
        "--tile-dims", help="Map of tile size matmul dims", type=str, default="mnk"
    )
    candidate_gen_args.add_argument(
        "--prefetch-shared-memory-options",
        type=lambda t: [s.strip().lower() == "true" for s in t.split(",")],
        default=[True],
        help="Comma-separated list of allowed values for the prefetch_shared_memory pipeline option. Possible values: [True, False]",
    )
    candidate_gen_args.add_argument(
        "--no-reduce-shared-memory-bank-conflicts-options",
        type=lambda t: [s.strip().lower() == "true" for s in t.split(",")],
        default=[None],
        help="Comma-separated list of allowed values for the no_reduce_shared_memory_bank_conflicts pipeline option. Possible values: [True, False]",
    )
    candidate_gen_args.add_argument(
        "--waves-per-eu-options",
        type=lambda t: [int(s) for s in t.split(",")],
        default=[2],
        help="Comma-separated list of allowed values for the waves_per_eu config option. Possible values: Any positive integer value",
    )
    candidate_gen_args.add_argument(
        "--codegen-pipeline",
        choices=[x.value for x in CodegenPipelines],
        default=CodegenPipelines.llvmgpu_vector_distribute,
        help="Codegen pipeline to tune for",
    )
    candidate_gen_args.add_argument(
        "--starter-td-spec",
        type=Path,
        default=None,
        help=(
            "Path to a starter TD spec file to merge with tuning spec files. "
            "Enables incremental merging of td specs across dispatches when tuning real models. "
            "Candidate specs take precedence in case of conflicts; the starter spec is excluded "
            "if fully covered."
        ),
    )

    if allow_unknown:
        args, _ = parser.parse_known_args()
        return args
    return parser.parse_args()


def setup_logging(args: argparse.Namespace, path_config: PathConfig) -> logging.Logger:
    log_file_name = f"autotune_{args.input_file.stem}.log"
    run_log_path = path_config.base_dir / log_file_name
    path_config.set_run_log(run_log_path)

    # Create file handler for logging to a file.
    if path_config.run_log is None:
        raise
    file_handler = logging.FileHandler(path_config.run_log)
    file_handler.setLevel(logging.DEBUG)

    # Create stream handler for logging to the console (only warnings and higher).
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Create a formatter that dynamically adds [levelname] for ERROR and WARNING.
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                return f"{record.message}"
            else:
                return f"[{record.levelname}] {record.message}"

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = CustomFormatter()

    # Set formatters to handlers.
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Configure the root logger.
    logging.basicConfig(
        level=logging.DEBUG,  # Set the root logger to the lowest level.
        handlers=[file_handler, console_handler],
    )

    # If verbose flag is set, add a console handler for INFO level and higher.
    if args.verbose:
        verbose_console_handler = logging.StreamHandler()
        verbose_console_handler.setLevel(logging.DEBUG)
        verbose_console_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(verbose_console_handler)

    # config logger in candidate_gen.py.
    tune_logger = logging.getLogger("tune")
    tune_logger.setLevel(logging.DEBUG)

    # Log all arguments.
    logging.debug(f"Input Arguments:")
    for arg, value in vars(args).items():
        logging.debug(f"{arg}: {value}")

    return logging.getLogger()


def handle_error(
    condition: bool,
    msg: str,
    level: int = logging.ERROR,
    error_type: Type[BaseException] = Exception,
    exit_program: bool = False,
) -> None:
    """If meets the condition, handles errors with logging and optional program exit"""
    if not condition:
        return

    # Log the message with the specified level.
    if level == logging.CRITICAL:
        logging.critical(msg)
        raise error_type(msg)
    if level == logging.ERROR:
        logging.error(msg)
        raise error_type(msg)
    elif level == logging.WARNING:
        logging.warning(msg)
    elif level == logging.INFO:
        logging.info(msg)
    elif level == logging.DEBUG:
        logging.debug(msg)
    else:
        raise ValueError(
            "Invalid logging level specified: choose from logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG"
        )

    if exit_program:
        sys.exit(1)


def init_worker_context(queue: multiprocessing.Queue) -> None:
    """Assign a static index to current process as the worker ordinal, and specify the device indices to be used"""
    global worker_id, device_id

    worker_id, device_id = queue.get()


def create_worker_context_queue(device_ids: list[str]) -> queue.Queue[tuple[int, int]]:
    """Create queue contains Worker ID and Device ID for worker initialization"""
    worker_contexts_queue = multiprocessing.Manager().Queue()
    for worker_id, device_id in enumerate(device_ids):
        worker_contexts_queue.put((worker_id, device_id))

    return worker_contexts_queue


def flatten_nested_td_spec(td_spec_str: str, output_path: Path) -> None:
    iree_opt = ireec.binaries.find_tool("iree-opt")  # type: ignore
    assert iree_opt, "iree-opt tool not found"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "tmp_input.mlir")
        with open(input_path, "w") as f:
            f.write(td_spec_str)

        link_result = subprocess.run(
            [
                iree_opt,
                "--iree-codegen-link-tuning-specs",
                input_path,
                "-o",
                output_path,
            ],
            capture_output=True,
            text=True,
        )

        if link_result.returncode != 0:
            raise RuntimeError(f"iree-opt failed: {link_result.stderr}")


def run_iree_compile_command(compile_pack: CompilePack) -> Optional[int]:
    candidate_tracker = compile_pack.candidate_tracker
    assert candidate_tracker.spec_path, "expected candidate spec path"

    if candidate_tracker.td_spec_str is not None:
        flatten_nested_td_spec(
            candidate_tracker.td_spec_str,
            candidate_tracker.spec_path,
        )

    # Compile to vmfb.
    td_spec_path = candidate_tracker.spec_path.as_posix()
    logging.debug(
        f"Compiling candidate {candidate_tracker.candidate_id} with spec: {td_spec_path}"
    )
    assert candidate_tracker.compiled_vmfb_path, "expected output vmfb path"
    output_path = candidate_tracker.compiled_vmfb_path.as_posix()
    crash_dump_path = f"{output_path}.crash_report.mlir"
    assert candidate_tracker.mlir_path, "expected input mlir file path"
    input_file = candidate_tracker.mlir_path.as_posix()
    iree_compile = ireec.binaries.find_tool("iree-compile")  # type: ignore
    assert iree_compile, "iree-compile tool not found"
    compile_command = [
        iree_compile,
        input_file,
        f"-o={output_path}",
        f"--mlir-pass-pipeline-crash-reproducer={crash_dump_path}",
        f"--iree-codegen-tuning-spec-path={td_spec_path}",
    ]
    compile_command += compile_pack.iree_compile_flags
    result = candidate_gen.run_command(
        candidate_gen.RunPack(
            command=compile_command,
            check=False,
            timeout_seconds=compile_pack.iree_compile_timeout,
        )
    )

    # We need to check if the output vmfb exists as iree-compile returns a success
    # status code when crash reproducers are dumped.
    output_vmfb_exists = candidate_tracker.compiled_vmfb_path.is_file()
    if result.process_res is None or result.is_timeout or not output_vmfb_exists:
        return None
    return candidate_tracker.candidate_id


def run_iree_benchmark_module_command(benchmark_pack: BenchmarkPack):
    candidate_tracker = benchmark_pack.candidate_tracker
    candidate_id = candidate_tracker.candidate_id

    # Load the candidate's vmfb and create vm_module.
    vmfb_path = candidate_tracker.compiled_vmfb_path
    assert vmfb_path is not None, "expected compiled_vmfb_path"
    with open(vmfb_path, "rb") as f:
        vmfb_buffer = f.read()

    vm_instance = ireert.VmInstance()
    vm_module = ireert.VmModule.copy_buffer(vm_instance, vmfb_buffer)

    # Parse the flags passed from the tuning client and create a kwargs dict
    # for the benchmark_module function.
    extra_flags = {}
    func_name = None
    inputs = []
    for flag in benchmark_pack.iree_benchmark_module_flags:
        assert flag[:2] == "--", "iree_benchmark_module_flags should begin with '--'"
        split_key_value = flag[2:].split("=")
        assert (
            len(split_key_value) >= 1
        ), "iree_benchmark_module_flags should have the format --<key>=<value>"
        key = split_key_value[0]
        value = "=".join(split_key_value[1:])
        # Allow the tuning client to pass `--function=@func_name`.
        if key == "function":
            func_name = value
            continue
        # Special handling for `--input`, since it can be passed many times.
        if key == "input":
            inputs.append(value)
            continue
        # Other flags become normal kwargs.
        extra_flags[key] = value

    # Benchmark the module.
    try:
        timeout = benchmark_pack.benchmark_timeout
        benchmark_results = ireert.benchmark.benchmark_module(
            vm_module,
            entry_function=func_name,
            inputs=inputs,
            device=device_id,
            timeout=timeout,
            **extra_flags,
        )
    except ireert.benchmark.BenchmarkTimeoutError as e:
        logging.info(
            f"Benchmark of candidate {candidate_id} timed out after {timeout:.5f} seconds."
        )
        return BenchmarkResult(
            candidate_id=candidate_id,
            time=math.inf,
            device_id=str(device_id),
        )

    times = []
    for benchmark_result in benchmark_results:
        benchmark_name = benchmark_result.benchmark_name
        # With multiple benchmark results, there will be `real_time_mean`, but
        # not with single iteration benchmark results, so ignore the mean time
        # and compute the mean of `real_time`, since the number of iterations
        # is up to the tuning client.
        if benchmark_name.split("/")[-1] == "real_time":
            time_and_unit = benchmark_result.time.split(" ")
            assert (
                len(time_and_unit) == 2
            ), "expected the benchmark time to be the time and unit separated by a space."
            time_us = unit_to_microseconds(
                real_time=float(time_and_unit[0]),
                time_unit=time_and_unit[1],
            )
            times.append(time_us)

    # If there are no times, then benchmarking failed at runtime. Record the
    # time as math.inf.
    if len(times) == 0:
        return BenchmarkResult(
            candidate_id=candidate_id,
            time=math.inf,
            device_id=str(device_id),
        )

    mean_benchmark_time = sum(times) / float(len(times))
    logging.debug(
        f"Benchmark time of candidate {candidate_id}: {mean_benchmark_time:.2f} us"
    )
    return BenchmarkResult(
        candidate_id=candidate_id,
        time=mean_benchmark_time,
        device_id=str(device_id),
    )


def multiprocess_progress_wrapper(
    num_worker: int,
    task_list: list,
    function: Callable,
    initializer: Optional[Callable] = None,
    initializer_inputs: Optional[Iterable[Any]] = None,
    time_budget: Optional[common.TimeBudget] = None,
) -> list[Any]:
    """Wrapper of multiprocessing pool and progress bar"""
    results = []
    initializer_inputs = initializer_inputs or ()

    # Create a multiprocessing pool.
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with multiprocessing.Pool(
        num_worker, initializer, initializer_inputs
    ) as worker_pool:
        signal.signal(signal.SIGINT, sigint_handler)
        # Use tqdm to create a progress bar.
        with tqdm(total=len(task_list)) as pbar:
            try:
                # Use imap_unordered to asynchronously execute the worker function on each task.
                for result in worker_pool.imap_unordered(function, task_list):
                    results.append(result)
                    pbar.update(1)  # Update progress bar.
                    # If time limit is reached, stop progress wrapper.
                    if time_budget is not None and time_budget.expired():
                        logging.warning(
                            f"Time limit reached, total {len(results)} results collected"
                        )
                        worker_pool.terminate()
                        worker_pool.join()
                        return results
            except KeyboardInterrupt:
                # If Ctrl+C is pressed, terminate all child processes.
                worker_pool.terminate()
                worker_pool.join()
                sys.exit(1)  # Exit the script.

    return results


def calculate_md5(file_path: Path) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def find_collisions(
    hash_list: list[tuple[int, str]]
) -> tuple[bool, list[tuple[str, list[int]]]]:
    """
    Detect hash value collisions
    Take input list of candidate index numbers and hash value strings: ex. [(1, 'abc'), (2, 'def'), (3, 'abc')]
    Return collision boolean value and list of unique hash values along with their corresponding indices: ex. [('abc', [1,3]), ('def', [2])]
    """
    hash_count: dict[str, list[int]] = {}

    # Count occurrences of each hash_val.
    for index, hash_val in hash_list:
        if hash_val in hash_count:
            hash_count[hash_val].append(index)
        else:
            hash_count[hash_val] = [index]

    # Prepare output for all hash values.
    hash_values = [(hash_val, indices) for hash_val, indices in hash_count.items()]

    # Determine if there are collisions.
    collisions_exist = any(len(indices) > 1 for hash_val, indices in hash_count.items())

    return collisions_exist, hash_values


def get_iree_codegen_pipeline(pipeline: CodegenPipelines):
    match pipeline:
        case CodegenPipelines.llvmgpu_vector_distribute:
            return iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
        case CodegenPipelines.llvmgpu_tile_and_fuse:
            return iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
        case _:
            assert False, "unexpected codegen pipeline"


def generate_candidate_specs(
    args: argparse.Namespace,
    path_config: PathConfig,
    tuning_client: TuningClient,
) -> list[int]:
    """Generate candidate transform dialect specs for tuning. Returns the list of candidate indexes"""
    logging.debug("generate_candidate_specs()")

    path_config.specs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.input_file, path_config.template_mlir)
    tune_logger = logging.getLogger("tune")

    # Generate transform dialect specs.
    try:
        # Strip compilation info before generating td_specs, since the generated
        # td_specs can end up matching against the compilation info from the
        # source mlir.
        mlir_text = candidate_gen.strip_compilation_info(path_config.template_mlir)
        mlir_module = dispatch_parser.parse_mlir(mlir_text, tuning_client.tuner_context)
        logging.debug("Captured messages from candidate_gen.py:")
        pipeline_options_search_space = dispatch_constraints.PipelineOptionsSearchSpace(
            prefetch_shared_memory=args.prefetch_shared_memory_options,
            no_reduce_shared_memory_bank_conflicts=args.no_reduce_shared_memory_bank_conflicts_options,
        )
        starter_td_spec: Optional[ir.Module] = None
        if args.starter_td_spec:
            with open(args.starter_td_spec, "r") as f:
                starter_td_spec = ir.Module.parse(f.read())

        dispatch_tuner = candidate_gen.set_dispatch_tuner(
            input_module=mlir_module, tuner_ctx=tuning_client.tuner_context
        )
        if not dispatch_tuner:
            tune_logger.warning(
                "Failed to set up dispatch tuner. No candidates will be generated."
            )
            return []

        tuning_client.target_info = common.get_target_info(mlir_module)
        assert tuning_client.target_info, "Failed to query target info."
        solutions_iter = candidate_gen.generate_solutions(
            dispatch_tuner=dispatch_tuner,
            target_info=tuning_client.target_info,
            tuner_context=tuning_client.tuner_context,
            num_subgroups=args.num_subgroups,
            allowed_waves_per_eu=args.waves_per_eu_options,
            pipeline_options_search_space=pipeline_options_search_space,
            codegen_pipeline=get_iree_codegen_pipeline(args.codegen_pipeline),
        )

        if args.enable_random_seed:
            random.seed()
        else:
            random.seed(args.search_space_shuffle_seed)

        solutions = list(solutions_iter)
        knobs: list[Optional[common.KnobAssignment]] = [
            dispatch_tuner.get_knob_assignment(s) for s in solutions
        ]

        sorted_order = candidate_ordering.reorder_assignments(
            knobs=knobs,
            strategy=args.candidate_order,
            target_info=tuning_client.target_info,
        )
        solutions = [solutions[i] for i in sorted_order] if sorted_order else solutions
        solutions = solutions[: args.num_candidates]

        config_specs: list[ir.Module] = candidate_gen.generate_configs_and_td_specs(
            dispatch_tuner=dispatch_tuner,
            input_module=mlir_module,
            solutions=solutions,
        )

        # Total number of configs = candidates generated + baseline.
        assert len(config_specs) == len(solutions) + 1

        tuning_client.tuning_records = (
            candidate_ordering.build_tuning_records_from_order(knobs, sorted_order)
        )

        knob_assignments = [dispatch_tuner.get_knob_assignment(s) for s in solutions]
        logging.debug("candidate_gen.py ends")
        handle_error(
            condition=(len(solutions) <= 1), msg="Failed to generate any candidates"
        )

        # Create candidate trackers.
        candidates = []
        for candidate_num, (spec, knob) in enumerate(
            zip(config_specs, knob_assignments)
        ):
            candidates.append(candidate_num)
            # Move the specs to the canonical path_config location.
            spec_path = path_config.specs_dir / path_config.get_candidate_spec_filename(
                candidate_num
            )

            td_spec_str: Optional[str] = None
            if starter_td_spec is not None:
                td_specs: list[ir.Module] = [spec, starter_td_spec]
                # Only log duplicate matchers during the first iteration.
                td_specs_to_link = common.determine_td_specs_to_link(
                    td_specs,
                    log_duplicates=(candidate_num == 0),
                )

                if len(td_specs_to_link) != 1:
                    td_spec_str = str(
                        common.combine_tuning_specs(
                            tuning_client.tuner_context, td_specs_to_link
                        )
                    )

            with open(spec_path, "w") as f:
                # Write the module with local scope so that compilation info
                # attributes are inlined. This makes it easier to split up the
                # TD spec and combine with other specs after tuning.
                local_scope_spec_str: str = spec.operation.get_asm(use_local_scope=True)
                f.write(local_scope_spec_str)
            new_candidate = CandidateTracker(
                mlir_path=path_config.template_mlir,
                candidate_id=candidate_num,
                spec_path=spec_path,
                td_spec_str=td_spec_str,
                # No knob_assignment for baseline.
                knob_assignment=knob if candidate_num != 0 else None,
            )
            tuning_client.candidate_trackers.append(new_candidate)
    except Exception as e:
        logging.error("An error occurred during candidates generation: %s", str(e))
        # Capture and log debug messages from candidate_gen.py.
        tune_logger = logging.getLogger("tune_with_td")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                tune_logger.handlers.append(handler)
        tune_logger.exception("Error in candidate_gen.py:")
        raise

    logging.debug(f"Generated [{len(candidates) - 1}] candidates")
    return candidates


def get_compilation_success_rate(compiled_candiates: list[Optional[int]]) -> float:
    if not compiled_candiates:
        return 0.0
    successful_candidates = [c for c in compiled_candiates if c is not None]
    success_rate = float(len(successful_candidates)) / float(len(compiled_candiates))
    return success_rate


def collision_handler(index_hash_list: list[tuple[int, str]]) -> tuple[bool, list[int]]:
    """If a collision is found, generate a list of new indexes. If no collision, `unique_indexes = []`"""
    # Check if candidate produces tbe same .vmfb.
    collision_detected, hash_list = find_collisions(index_hash_list)
    unique_indexes: list[int] = []
    if not collision_detected:
        return collision_detected, unique_indexes

    # If a collision is detected, select the first one from the collided list.
    logging.warning("Collisions detected")
    for hash_val, indices in hash_list:
        if len(indices) != 1:
            logging.warning(f"Hash value '{hash_val}' collided at candidate {indices}.")
        unique_indexes.append(indices[0])

    return collision_detected, unique_indexes


def benchmark_candidates(
    candidate_indices: list[int],
    devices: list[str],
    tuning_client: TuningClient,
    timeout_reference=float,
    benchmark_time: Optional[float] = None,
) -> list[BenchmarkResult]:
    """
    Runs the benchmarking for a given list of candidate indices.
    """
    worker_context_queue = create_worker_context_queue(devices)

    benchmark_timeout = (
        timeout_reference
        if tuning_client.is_auto_iree_benchmark_timeout()
        else tuning_client.get_iree_benchmark_timeout_s()
    )

    task_list = [
        BenchmarkPack(
            iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
            benchmark_timeout=benchmark_timeout,
            candidate_tracker=tuning_client.candidate_trackers[idx],
        )
        for idx in candidate_indices
    ]

    # Perform benchmarking.
    return multiprocess_progress_wrapper(
        num_worker=len(devices),
        task_list=task_list,
        function=run_iree_benchmark_module_command,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
        time_budget=common.TimeBudget.for_minutes(benchmark_time),
    )


def benchmark_baseline(
    devices: list[str],
    tuning_client: TuningClient,
    candidate_tracker: CandidateTracker,
) -> tuple[list[BenchmarkResult], float]:

    global worker_id, device_id

    baseline_results = list()

    running_time_s: list[float] = []
    # Use tqdm to create a progress bar.
    with tqdm(total=len(devices)) as pbar:
        try:
            worker_id = 0
            for device_id_ in devices:
                device_id = device_id_
                benchmark_start_timestamp = time.perf_counter()
                result = run_iree_benchmark_module_command(
                    BenchmarkPack(
                        iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
                        benchmark_timeout=tuning_client.get_iree_benchmark_timeout_s(),
                        candidate_tracker=candidate_tracker,
                    )
                )
                elapsed_s = time.perf_counter() - benchmark_start_timestamp
                running_time_s.append(elapsed_s)
                baseline_results.append(result)
                pbar.update(1)  # Update progress bar.
        except KeyboardInterrupt:
            # If Ctrl+C is pressed, terminate all child processes.
            sys.exit(1)  # Exit the script.
    logging.debug(
        f"Baseline benchmarking subprocess running time list is: {running_time_s}\n"
    )
    subprocess_timeout_reference = max(running_time_s) * DEFAULT_TIMEOUT_MUL
    return baseline_results, subprocess_timeout_reference


class BaselineResultHandler:
    def __init__(self) -> None:
        # Maps device IDs to a list of `BenchmarkResult`.
        self.device_baseline_results: dict[str, list[BenchmarkResult]] = defaultdict(
            list
        )

    def add_run(self, results: list[BenchmarkResult]) -> None:
        if not BaselineResultHandler.are_baseline_devices_unique(results):
            logging.warning(
                "Duplicate device IDs detected in the first baseline results."
            )
        for result in results:
            self.device_baseline_results[result.device_id].append(result)

    @staticmethod
    def are_baseline_devices_unique(results: list[BenchmarkResult]) -> bool:
        return len(results) == len(set(result.device_id for result in results))

    def get_valid_time_us(self, device_id: str) -> list[float]:
        return [
            result.time
            for result in self.device_baseline_results.get(device_id, [])
            if result.is_valid()
        ]

    def get_average_result_us(self, device_id: str) -> Optional[float]:
        valid_times = self.get_valid_time_us(device_id)
        if valid_times:
            return sum(valid_times) / len(valid_times)
        return None

    def detect_regressions(
        self,
        baseline_results: list[BenchmarkResult],
        threshold: float = 1.03,
    ) -> list[str]:
        """
        Returns a list of device IDs where performance regressions were detected.
        A performance regression is defined as a baseline time from 'baseline_results'
        for a device that exceeds the stored average baseline time for that device by
        a factor greater than the specified 'threshold'.
        """
        regressions = []
        for result in baseline_results:
            if not result.is_valid():
                continue

            baseline_avg = self.get_average_result_us(result.device_id)
            if baseline_avg is not None and result.time > baseline_avg * threshold:
                regressions.append(result.device_id)

        return regressions

    def is_valid(self) -> bool:
        """
        Check if there are any valid finite baseline time recorded.
        Returns True iff at least one valid (finite) baseline time was recorded.
        """
        return any(
            self.get_valid_time_us(device_id)
            for device_id in self.device_baseline_results
        )

    def is_valid_for_device(self, device_id: str) -> bool:
        return len(self.get_valid_time_us(device_id)) != 0

    def get_fallback_baseline(self) -> Optional[float]:
        if not self.is_valid():
            logging.warning("No valid baseline times available.")
            return None

        # Calculate the fallback baseline as the average of all valid times across devices.
        valid_baseline_times = [
            result.time
            for results in self.device_baseline_results.values()
            for result in results
            if result.is_valid()
        ]
        assert valid_baseline_times
        return sum(valid_baseline_times) / len(valid_baseline_times)

    def is_better_than_baseline(self, candidate_results: list[BenchmarkResult]) -> bool:
        """
        Returns True if any candidate runs faster than the baseline.

        Uses the device-specific average baseline if available,
        otherwise falls back to the overall average baseline time.
        """
        fallback_baseline = self.get_fallback_baseline()
        if not fallback_baseline:
            return False
        for candidate in candidate_results:
            baseline_avg_us = self.get_average_result_us(candidate.device_id)
            if baseline_avg_us is None:
                baseline_avg_us = fallback_baseline
            assert baseline_avg_us
            if candidate.time < baseline_avg_us:
                return True
        return False

    def get_candidates_ordered_by_speedup(
        self,
        candidate_results: list[BenchmarkResult],
        prune_slow_candidates: bool = False,
    ) -> list[tuple[BenchmarkResult, float]]:
        """
        Returns a list of tuples (BenchmarkResult, speedup) sorted in ascending order based on speedup
        or raw runtime.

        If no valid baseline times are available across all devices, candidates are sorted based on
        their raw runtime. A placeholder speedup value of 1.0 is assigned to each candidate.

        If valid baseline times exist, speedup is defined as the ratio of the candidate's runtime to
        the average baseline time for the corresponding device as:

            speedup = candidate_runtime / avg_baseline_time (or fallback_baseline)

        If no valid baseline times are available for a specific device, the fallback baseline is used.
        The fallback baseline is the average of all valid baseline times across devices.

        Args:
            candidate_results: List of benchmark results to sort.
            prune_slow_candidates: If True and all candidates are slower than baseline,
                returns empty list. Otherwise, returns all sorted candidates regardless.
        """
        # Check if all candidates are slower than baseline and should be pruned.
        if prune_slow_candidates and not self.is_better_than_baseline(
            candidate_results
        ):
            return []

        if not self.is_valid():
            logging.warning("No valid baseline times available.")
            # Use the candidate time directly when no baselines are available.
            return sorted(
                [(candidate, 1.0) for candidate in candidate_results],
                key=lambda x: x[0].time,
            )

        fallback_baseline = self.get_fallback_baseline()
        candidates_with_speedup = []
        for candidate in candidate_results:
            baseline_avg_us = self.get_average_result_us(candidate.device_id)
            if baseline_avg_us is None:
                baseline_avg_us = fallback_baseline
                assert baseline_avg_us
            speedup = candidate.time / baseline_avg_us
            candidates_with_speedup.append((candidate, speedup))
        return sorted(candidates_with_speedup, key=lambda x: x[1])


def compile(
    args: argparse.Namespace,
    path_config: PathConfig,
    candidates: list[int],
    tuning_client: TuningClient,
    input_file: Optional[Path] = None,
) -> list[int]:
    logging.debug("compile()")

    if not candidates:
        logging.warning("No model candidates to compile.")
        return []

    # If `input_file` is not None, then replace the currently tracked template
    # with the passed input mlir file.
    if input_file is not None:
        shutil.copy(input_file, path_config.template_mlir)

    # Strip compilation info and root_op attribute from the source and save
    # the stripped IR, since the TD specs do not expect these attributes.
    stripped_mlir = candidate_gen.strip_compilation_info(path_config.template_mlir)
    context = tuning_client.tuner_context.mlir_ctx
    stripped_module = ir.Module.parse(stripped_mlir, context=context)
    candidate_gen.strip_root_op_attr(stripped_module)
    stripped_mlir = str(stripped_module)
    with open(path_config.template_mlir, "w") as f:
        f.write(stripped_mlir)

    # Set the source and output file paths for compilation of each candidate.
    path_config.compiled_dir.mkdir(parents=True, exist_ok=True)
    for i in candidates:
        tuning_client.tuning_records[i].to_compile = True
        vmfb_file_name = path_config.get_candidate_vmfb_filename(
            tuning_client.candidate_trackers[i].candidate_id
        )
        vmfb_path = path_config.compiled_dir / vmfb_file_name
        tuning_client.candidate_trackers[i].compiled_vmfb_path = vmfb_path
        tuning_client.candidate_trackers[i].mlir_path = path_config.template_mlir
    tuning_client.candidate_trackers[0].mlir_path = path_config.template_mlir

    # Run compilation for all candidates.
    task_list = [
        CompilePack(
            iree_compile_flags=tuning_client.get_iree_compile_flags(),
            iree_compile_timeout=tuning_client.get_iree_compile_timeout_s(),
            candidate_tracker=tuning_client.candidate_trackers[i],
        )
        for i in candidates
    ]
    if 0 not in candidates:
        task_list.append(
            CompilePack(
                iree_compile_flags=tuning_client.get_iree_compile_flags(),
                iree_compile_timeout=tuning_client.get_iree_compile_timeout_s(),
                candidate_tracker=tuning_client.candidate_trackers[0],
            )
        )
    num_worker = min(args.max_cpu_workers, len(task_list))
    compiled_candidates = multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_iree_compile_command
    )
    success_rate = get_compilation_success_rate(compiled_candidates)
    logging.debug(
        f"Successfully compiled [{len(compiled_candidates)}] candidates. Success rate: {success_rate:.2f}"
    )
    compiled_candidates = [c for c in compiled_candidates if c is not None]

    # Remove duplicate vmfbs from the candidate list.
    compiled_candidate_hashes = []
    for candidate_id in compiled_candidates:
        tuning_client.tuning_records[candidate_id].compile_status = True
        candidate_vmfb = tuning_client.candidate_trackers[
            candidate_id
        ].compiled_vmfb_path
        hash_val = calculate_md5(candidate_vmfb)
        compiled_candidate_hashes.append((candidate_id, hash_val))
    collision_detected, unique_compiled_candidates = collision_handler(
        compiled_candidate_hashes
    )
    if collision_detected:
        compiled_candidates = unique_compiled_candidates

    logging.debug(f"Produced [{len(compiled_candidates)}] unique vmfbs")

    # libtuner assigns candidate_ids in generate_candidate_specs() based
    # on CandidateOrderKind.
    # Parallel compilation may change the order of compiled candidates.
    # Sorting by candidate_id restores the original order for consistent
    # benchmarking.
    compiled_candidates.sort()

    return compiled_candidates


def benchmark(
    args: argparse.Namespace,
    compiled_candidates: list[int],
    tuning_client: TuningClient,
    num_candidates: Optional[int] = None,
    benchmark_time: Optional[float] = None,
):
    logging.debug("benchmark()")
    if len(compiled_candidates) == 0:
        logging.warning("No candidates to benchmark.")
        return []

    # Benchmarking baselines on each involved device.
    baseline_tracker = tuning_client.candidate_trackers[0]
    first_baseline_result, subprocess_timeout_reference = benchmark_baseline(
        devices=args.devices,
        tuning_client=tuning_client,
        candidate_tracker=baseline_tracker,
    )
    baseline_handler = BaselineResultHandler()
    baseline_handler.add_run(first_baseline_result)
    if not baseline_handler.is_valid():
        logging.warning("Baseline run failed.")

    if tuning_client.is_auto_iree_benchmark_timeout():
        logging.info(
            f"Smart candidate benchmark timeout is set to {subprocess_timeout_reference:.2f}s"
        )
    candidate_indices = [i for i in compiled_candidates if i != 0]
    for i, idx in enumerate(candidate_indices, start=1):
        tuning_client.tuning_records[idx].benchmark_queue_position = i
        tuning_client.tuning_records[idx].to_benchmark = True

    candidate_results = benchmark_candidates(
        candidate_indices=candidate_indices,
        devices=args.devices,
        tuning_client=tuning_client,
        timeout_reference=subprocess_timeout_reference,
        benchmark_time=benchmark_time,  # Only candidate benchmark has time limit.
    )

    for res in candidate_results:
        c_id = res.candidate_id
        res_time = res.time
        tuning_client.tuning_records[c_id].benchmark_device_id = res.device_id
        if res_time == math.inf:
            continue
        tuning_client.tuning_records[c_id].benchmark_status = True
        tuning_client.tuning_records[c_id].benchmark_time_us = round(res_time, 2)

    second_baseline_result, _ = benchmark_baseline(
        devices=args.devices,
        tuning_client=tuning_client,
        candidate_tracker=baseline_tracker,
    )

    regression_devices = baseline_handler.detect_regressions(second_baseline_result)
    if regression_devices:
        logging.warning(
            f"Performance regressions detected for the following devices: {', '.join(regression_devices)}."
        )
    baseline_handler.add_run(second_baseline_result)

    if not baseline_handler.is_valid():
        logging.warning("Baseline run failed.")

    if not baseline_handler.is_better_than_baseline(candidate_results):
        logging.warning("All candidates are slower than the baseline.")

    all_candidates_with_speedup = baseline_handler.get_candidates_ordered_by_speedup(
        candidate_results,
        prune_slow_candidates=tuning_client.should_prune_slower_candidates(),
    )

    # Best candidate gets rank 1.
    for i, handler_res in enumerate(all_candidates_with_speedup, start=1):
        benchmark_res, speedup = handler_res
        cid, _, device_id = benchmark_res
        baseline_res = baseline_handler.get_average_result_us(device_id)
        tuning_client.tuning_records[cid].baseline_benchmark_time_us = (
            round(baseline_res, 2) if baseline_res else None
        )
        tuning_client.tuning_records[cid].benchmark_speedup = round(speedup, 5)
        tuning_client.tuning_records[cid].benchmark_rank_order = i

    top_candidates_with_speedup = (
        all_candidates_with_speedup[:num_candidates]
        if num_candidates
        else all_candidates_with_speedup
    )

    if baseline_handler.is_valid():
        for candidate, speedup in top_candidates_with_speedup:
            time_us = candidate.time
            candidate_id = candidate.candidate_id
            percentage_of_baseline = speedup * 100
            logging.info(
                f"Candidate {candidate_id} time: {time_us:.2f} us "
                f"({percentage_of_baseline:.1f}% of baseline)"
            )
    else:
        for candidate, _ in top_candidates_with_speedup:
            time_us = candidate.time
            candidate_id = candidate.candidate_id
            logging.info(f"Candidate {candidate_id} time: {time_us:.2f} us")

    return [candidate.candidate_id for candidate, _ in top_candidates_with_speedup]
