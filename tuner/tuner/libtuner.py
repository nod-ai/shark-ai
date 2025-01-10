# Copyright 2024 Advanced Micro Devices, Inc
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
import iree.runtime as ireert  # type: ignore
import iree.compiler as ireec  # type: ignore
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
from . import candidate_gen
from . import dispatch_parser
from .op_matchers import *
from .common import *
from .dispatch_constraints import *


# Default values for num_candidates and devices, change it as needed
DEFAULT_NUM_CANDIDATES = 2048
DEFAULT_DEVICE_LIST = ["hip://0"]

# Default values for max number of workers
DEFAULT_MAX_CPU_WORKERS = (
    multiprocessing.cpu_count() // 2
)  # the actual amount of worker that will be generated = min(max_cpu_workers, len(task_list))

# Declare global variables at the module level for multiprocessing
worker_id = None
device_id = None

# Declare special symbols for libtuner to search and locate
DEVICE_ID_PLACEHOLDER = "!DEVICE_ID!"


@dataclass
class CandidateTracker:
    candidate_id: int
    mlir_path: Optional[Path] = None
    compiled_vmfb_path: Optional[Path] = None
    spec_path: Optional[Path] = None


@dataclass()
class PathConfig:
    # Dynamic paths
    base_dir: Path = field(init=False)
    template_mlir: Path = field(init=False)
    candidates_dir: Path = field(init=False)
    compiled_dir: Path = field(init=False)
    specs_dir: Path = field(init=False)

    # To be set outside of class
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

    def _set_run_log(self, run_log: Path):
        object.__setattr__(self, "run_log", run_log)

    def get_candidate_spec_filename(self, candidate_id: int) -> str:
        return f"{candidate_id}_spec.mlir"

    def get_candidate_vmfb_filename(self, candidate_id: int) -> str:
        return f"{candidate_id}.vmfb"


class TuningClient(ABC):
    def __init__(self, tuner_context: TunerContext):
        self.tuner_context = tuner_context

    @abstractmethod
    def get_iree_compile_flags(self) -> list[str]:
        pass

    @abstractmethod
    def get_iree_compile_timeout_s(self) -> int:
        pass

    @abstractmethod
    def get_iree_benchmark_module_flags(self) -> list[str]:
        pass

    @abstractmethod
    def get_benchmark_timeout_s(self) -> int:
        pass


@dataclass
class CompilePack:
    iree_compile_flags: list[str]
    iree_compile_timeout: int
    candidate_tracker: CandidateTracker


@dataclass
class BenchmarkPack:
    iree_benchmark_module_flags: list[str]
    benchmark_timeout: int
    candidate_tracker: CandidateTracker


@dataclass
class BenchmarkResult:
    candidate_id: int
    time: float
    device_id: str


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
    Extract all available devices on the user's machine for the provided drivers
    Only the user provided drivers will be queried
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
) -> argparse.Namespace:
    parser = initial_parser
    if parser is None:
        parser = argparse.ArgumentParser(description="Autotune script")

    # Required arguments
    required_args = parser.add_argument_group("Required Options")
    required_args.add_argument(
        "input_file", type=Path, help="Path to the input benchmark file (.mlir)"
    )

    # General options
    general_args = parser.add_argument_group("General Options")
    general_args.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
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
        "--num-model-candidates",
        help="Maximum number of stage 2 candidates",
        type=int,
        default=50,
    )
    general_args.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not attempt to run any modules or initialize the IREE runtime",
    )

    # candidate_gen.tune() options
    candidate_gen_args = parser.add_argument_group("Candidate Generation Options")
    candidate_gen_args.add_argument(
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Number of candidates to be generated by candidate_gen.py (default: {DEFAULT_NUM_CANDIDATES})",
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
    general_args.add_argument(
        "--codegen-pipeline",
        choices=[x.value for x in CodegenPipelines],
        default=CodegenPipelines.llvmgpu_vector_distribute,
        help="Codegen pipeline to tune for",
    )

    return parser.parse_args()


def setup_logging(args: argparse.Namespace, path_config: PathConfig):
    log_file_name = f"autotune_{args.input_file.stem}.log"
    run_log_path = path_config.base_dir / log_file_name
    path_config._set_run_log(run_log_path)

    # Create file handler for logging to a file
    if path_config.run_log is None:
        raise
    file_handler = logging.FileHandler(path_config.run_log)
    file_handler.setLevel(logging.DEBUG)

    # Create stream handler for logging to the console (only warnings and higher)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Create a formatter that dynamically adds [levelname] for ERROR and WARNING
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                return f"{record.message}"
            else:
                return f"[{record.levelname}] {record.message}"

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = CustomFormatter()

    # Set formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set the root logger to the lowest level
        handlers=[file_handler, console_handler],
    )

    # If verbose flag is set, add a console handler for INFO level and higher
    if args.verbose:
        verbose_console_handler = logging.StreamHandler()
        verbose_console_handler.setLevel(logging.DEBUG)
        verbose_console_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(verbose_console_handler)

    # config logger in candidate_gen.py
    tune_logger = logging.getLogger("tune")
    tune_logger.setLevel(logging.DEBUG)

    # Log all arguments
    logging.debug(f"Input Arguments:")
    for arg, value in vars(args).items():
        tune_logger.info(f"{arg}: {value}")


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

    # Log the message with the specified level
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
    """Assign a static index to current process as the worker ordinal, and specify the device indice to be used"""
    global worker_id, device_id

    worker_id, device_id = queue.get()


def create_worker_context_queue(device_ids: list[int]) -> queue.Queue[tuple[int, int]]:
    """Create queue contains Worker ID and Device ID for worker initialization"""
    worker_contexts_queue = multiprocessing.Manager().Queue()
    for worker_id, device_id in enumerate(device_ids):
        worker_contexts_queue.put((worker_id, device_id))

    return worker_contexts_queue


def run_iree_compile_command(compile_pack: CompilePack) -> Optional[int]:
    candidate_tracker = compile_pack.candidate_tracker

    # Compile to vmfb.
    assert candidate_tracker.spec_path, "expected candidate spec path"
    td_spec_path = candidate_tracker.spec_path.as_posix()
    logging.debug(
        f"Compiling candidate {candidate_tracker.candidate_id} with spec: {td_spec_path}"
    )
    assert candidate_tracker.compiled_vmfb_path, "expected output vmfb path"
    output_path = candidate_tracker.compiled_vmfb_path.as_posix()
    crash_dump_path = f"{output_path}.crash_report.mlir"
    assert candidate_tracker.mlir_path, "expected input mlir file path"
    input_file = candidate_tracker.mlir_path.as_posix()
    iree_compile = ireec.binaries.find_tool("iree-compile")
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
            f"Benchmark of candidate {candidate_id} timed out after {timeout} seconds."
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
        f"Benchmark time of candidate {candidate_id}: {mean_benchmark_time:.2f}"
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
) -> list[Any]:
    """Wrapper of multiprocessing pool and progress bar"""
    results = []
    initializer_inputs = initializer_inputs or ()

    # Create a multiprocessing pool
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with multiprocessing.Pool(
        num_worker, initializer, initializer_inputs
    ) as worker_pool:
        signal.signal(signal.SIGINT, sigint_handler)
        # Use tqdm to create a progress bar
        with tqdm(total=len(task_list)) as pbar:
            try:
                # Use imap_unordered to asynchronously execute the worker function on each task
                for result in worker_pool.imap_unordered(function, task_list):
                    pbar.update(1)  # Update progress bar
                    results.append(result)
            except KeyboardInterrupt:
                # If Ctrl+C is pressed, terminate all child processes
                worker_pool.terminate()
                worker_pool.join()
                sys.exit(1)  # Exit the script

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

    # Count occurrences of each hash_val
    for index, hash_val in hash_list:
        if hash_val in hash_count:
            hash_count[hash_val].append(index)
        else:
            hash_count[hash_val] = [index]

    # Prepare output for all hash values
    hash_values = [(hash_val, indices) for hash_val, indices in hash_count.items()]

    # Determine if there are collisions
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
    candidate_trackers: list[CandidateTracker],
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
        pipeline_options_search_space = PipelineOptionsSearchSpace(
            prefetch_shared_memory=args.prefetch_shared_memory_options,
            no_reduce_shared_memory_bank_conflicts=args.no_reduce_shared_memory_bank_conflicts_options,
        )
        config_specs: list[ir.Module] = candidate_gen.generate_configs_and_td_specs(
            input_module=mlir_module,
            tuner_context=tuning_client.tuner_context,
            limit=args.num_candidates,
            num_subgroups=args.num_subgroups,
            allowed_waves_per_eu=args.waves_per_eu_options,
            pipeline_options_search_space=pipeline_options_search_space,
            codegen_pipeline=get_iree_codegen_pipeline(args.codegen_pipeline),
        )
        logging.debug("candidate_gen.py ends")
        handle_error(
            condition=(len(config_specs) <= 1), msg="Failed to generate any candidates"
        )

        # Create candidate trackers.
        candidates = []
        for candidate_num, spec in enumerate(config_specs):
            candidates.append(candidate_num)
            # Move the specs to the canonical path_config location.
            spec_path = path_config.specs_dir / path_config.get_candidate_spec_filename(
                candidate_num
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
            )
            candidate_trackers.append(new_candidate)
    except Exception as e:
        logging.error("An error occurred during candidates generation: %s", str(e))
        # Capture and log debug messages from candidate_gen.py.
        tune_logger = logging.getLogger("tune_with_td")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                tune_logger.handlers.append(handler)
        tune_logger.exception("Error in candidate_gen.py:")
        raise

    logging.info(f"Generated [{len(candidates) - 1}] candidates")
    return candidates


def collision_handler(index_hash_list: list[tuple[int, str]]) -> tuple[bool, list[int]]:
    """If a collision is found, generate a list of new indexes. If no collision, `unique_indexes = []`"""
    # Check if candidate produces tbe same .vmfb
    collision_detected, hash_list = find_collisions(index_hash_list)
    unique_indexes: list[int] = []
    if not collision_detected:
        return collision_detected, unique_indexes

    # If a collision is detected, select the first one from the collided list
    logging.warning("Collisions detected")
    for hash_val, indices in hash_list:
        if len(indices) != 1:
            logging.warning(f"Hash value '{hash_val}' collided at candidate {indices}.")
        unique_indexes.append(indices[0])

    return collision_detected, unique_indexes


def compile(
    args: argparse.Namespace,
    path_config: PathConfig,
    candidates: list[int],
    candidate_trackers: list[CandidateTracker],
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
        vmfb_file_name = path_config.get_candidate_vmfb_filename(
            candidate_trackers[i].candidate_id
        )
        vmfb_path = path_config.compiled_dir / vmfb_file_name
        candidate_trackers[i].compiled_vmfb_path = vmfb_path
        candidate_trackers[i].mlir_path = path_config.template_mlir
    candidate_trackers[0].mlir_path = path_config.template_mlir

    # Run compilation for all candidates.
    task_list = [
        CompilePack(
            iree_compile_flags=tuning_client.get_iree_compile_flags(),
            iree_compile_timeout=tuning_client.get_iree_compile_timeout_s(),
            candidate_tracker=candidate_trackers[i],
        )
        for i in candidates
    ]
    if 0 not in candidates:
        task_list.append(
            CompilePack(
                iree_compile_flags=tuning_client.get_iree_compile_flags(),
                iree_compile_timeout=tuning_client.get_iree_compile_timeout_s(),
                candidate_tracker=candidate_trackers[0],
            )
        )
    num_worker = min(args.max_cpu_workers, len(task_list))
    compiled_candidates = multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_iree_compile_command
    )
    compiled_candidates = [c for c in compiled_candidates if c is not None]
    success_rate = float(len(compiled_candidates)) / float(len(candidates))
    logging.info(
        f"Successfully compiled [{len(compiled_candidates)}] candidates. Success rate: {success_rate:.2f}"
    )

    # Remove duplicate vmfbs from the candidate list.
    compiled_candidate_hashes = []
    for candidate_id in compiled_candidates:
        candidate_vmfb = candidate_trackers[candidate_id].compiled_vmfb_path
        hash_val = calculate_md5(candidate_vmfb)
        compiled_candidate_hashes.append((candidate_id, hash_val))
    collision_detected, unique_compiled_candidates = collision_handler(
        compiled_candidate_hashes
    )
    if collision_detected:
        compiled_candidates = unique_compiled_candidates

    logging.info(f"Produced [{len(compiled_candidates)}] unique vmfbs")
    return compiled_candidates


def select_best_benchmark_results(
    candidate_results: list[BenchmarkResult],
    baseline_results: list[BenchmarkResult],
    num_candidates: Optional[int],
) -> list[BenchmarkResult]:
    filtered_candidate_results = [r for r in candidate_results if math.isfinite(r.time)]
    if len(filtered_candidate_results) == 0:
        logging.error("No successful candidate benchmarks.")
        return []
    fallback_baseline_time: Optional[float] = None
    filtered_baseline_results: list[BenchmarkResult] = []
    for r in baseline_results:
        if math.isfinite(r.time):
            filtered_baseline_results.append(r)
            fallback_baseline_time = r.time
        else:
            logging.warning(f"Baseline on device {r.device_id} failed.")
    if fallback_baseline_time is None:
        logging.warning(
            f"All baseline benchmarks failed. Baselines will not be used to select top candidates"
        )
    baseline_times_by_device = {}
    for r in filtered_baseline_results:
        baseline_times_by_device[r.device_id] = r.time

    # Select top candidates
    def get_speedup(result: BenchmarkResult) -> float:
        if result.device_id in baseline_times_by_device:
            return result.time / baseline_times_by_device[result.device_id]
        assert fallback_baseline_time is not None, "expected fallback_baseline_time"
        return result.time / fallback_baseline_time

    num_top_candidates = len(filtered_candidate_results)
    if num_candidates is not None:
        num_top_candidates = num_candidates

    # Sort by the speedup over baseline on the same device. If a device failed
    # the baseline benchmark, then use the fallback baseline. If there is no
    # successful baseline, then the best we can do is to sort by the actual
    # time.
    sorting_key = get_speedup
    if fallback_baseline_time is None:
        sorting_key = lambda result: result.time
    best_results = sorted(filtered_candidate_results, key=sorting_key)[
        :num_top_candidates
    ]
    logging.info(f"Selected top[{len(best_results)}]:")

    for r in best_results:
        if fallback_baseline_time is not None:
            speedup = f"{round(get_speedup(r) * 100, 2)}% of baseline"
        else:
            speedup = "baseline unavailable"
        logging.info(f"Candidate {r.candidate_id} time: {r.time:.2f} ({speedup})")
    return best_results


def benchmark(
    args: argparse.Namespace,
    path_config: PathConfig,
    compiled_candidates: list[int],
    candidate_trackers: list[CandidateTracker],
    tuning_client: TuningClient,
    num_candidates: Optional[int] = None,
):
    logging.debug("benchmark()")
    if len(compiled_candidates) == 0:
        logging.warning("No candidates to benchmark.")
        return []

    task_list = [
        BenchmarkPack(
            iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
            benchmark_timeout=tuning_client.get_benchmark_timeout_s(),
            candidate_tracker=candidate_trackers[i],
        )
        for i in compiled_candidates
        if i != 0
    ]
    worker_context_queue = create_worker_context_queue(args.devices)
    candidate_results: list[BenchmarkResult] = multiprocess_progress_wrapper(
        num_worker=len(args.devices),
        task_list=task_list,
        function=run_iree_benchmark_module_command,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
    )

    # Benchmarking baselines on each involved device.
    worker_context_queue = create_worker_context_queue(args.devices)
    baseline_task_list = [
        BenchmarkPack(
            iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
            benchmark_timeout=tuning_client.get_benchmark_timeout_s(),
            candidate_tracker=candidate_trackers[0],
        )
    ] * len(args.devices)
    baseline_results: list[BenchmarkResult] = multiprocess_progress_wrapper(
        num_worker=len(args.devices),
        task_list=baseline_task_list,
        function=run_iree_benchmark_module_command,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
    )

    best_results: list[BenchmarkResult] = select_best_benchmark_results(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        num_candidates=num_candidates,
    )

    top_candidates = [result.candidate_id for result in best_results]
    return top_candidates
