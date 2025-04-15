from iree.build.executor import FileNamespace, BuildAction, BuildContext, BuildFile
import os
import re
import urllib
import logging
import asyncio
from pathlib import Path
import struct
import threading
from typing import Optional, Union

import shortfin.array as sfnp
import shortfin as sf
from shortfin.interop.support.device_setup import get_selected_devices


def get_system_args(parser):
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["local-task", "hip", "amdgpu"],
        help="Device to serve on; e.g. local-task, hip. Same options as `iree-run-module --device` ",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        nargs="*",
        default=None,
        help="Device IDs visible to the system builder. Defaults to None (full visibility). Can be an index or a sf device id like amdgpu:0:0@0",
    )
    parser.add_argument(
        "--amdgpu_async_allocations",
        action="store_true",
        help="Enable asynchronous allocations for amdgpu device contexts.",
    )
    parser.add_argument(
        "--amdgpu_allow_device_reuse",
        action="store_true",
        help="Allows the same device to be used for each instance",
    )
    parser.add_argument(
        "--amdgpu_allocators",
        default=None,
        help="Allocator to use during VMFB invocation.",
    )


class SystemManager:
    def __init__(
        self,
        device: str = "local-task",
        device_ids: list[Union[str, int]] = None,
        async_allocs: bool = True,
        amdgpu_allow_device_reuse: bool = False,
        amdgpu_allocators: Optional[bool] = None,
        logger_name: str = __name__,
        shutdown_system: bool = True,
    ):
        self.logger = logging.getLogger(logger_name)

        self.shutdown_system = shutdown_system

        if any(x in device for x in ["local-task", "cpu"]):
            self.ls = sf.host.CPUSystemBuilder().create_system()
        elif any(x in device for x in ["hip", "amdgpu"]):
            if amdgpu_allocators is None:
                sb = sf.SystemBuilder(
                    system_type="amdgpu",
                    amdgpu_async_allocations=async_allocs,
                    amdgpu_allow_device_reuse=amdgpu_allow_device_reuse,
                )
            else:
                sb = sf.SystemBuilder(
                    system_type="amdgpu",
                    amdgpu_async_allocations=async_allocs,
                    amdgpu_allocators=amdgpu_allocators,
                    amdgpu_allow_device_reuse=amdgpu_allow_device_reuse,
                )
            if device_ids:
                sb.visible_devices = sb.available_devices
                sb.visible_devices = get_selected_devices(sb, device_ids)
            self.ls = sb.create_system()

        self.logger.info(f"Created local system with {self.ls.device_names} devices")
        # TODO: Come up with an easier bootstrap thing than manually
        # running a thread.
        self.t = threading.Thread(target=lambda: self.ls.run(self.run()))
        self.command_queue = self.ls.create_queue("command")
        self.command_writer = self.command_queue.writer()

    def start(self):
        self.logger.info("Starting system manager")
        self.t.start()

    def shutdown(self):
        self.logger.info("Shutting down system manager")
        self.command_queue.close()
        if self.shutdown_system:
            self.ls.shutdown()

    async def run(self):
        reader = self.command_queue.reader()
        while command := await reader():
            ...
        self.logger.info("System manager command processor stopped")


def convert_int_to_float(value: int, dtype: sfnp.DType) -> float:
    """Convert an `int` representation of a `float` value to float64.

    Args:
        value (int): Int representation of the float value.
        dtype (sfnp.DType): Shortfin dtype of the value.

    Raises:
        ValueError: Unsupported `dtype`.

    Returns:
        float: float64 representation of original value.
    """
    format_specs = {
        str(sfnp.float16): "<H",
    }
    format_spec = format_specs.get(str(dtype))
    if format_spec is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Convert the int value to bytes
    packed_val = struct.pack(format_spec, value)
    # Unpack the bytes to a float
    return struct.unpack("<e", packed_val)[0]


dtype_to_filetag = {
    "bfloat16": "bf16",
    "float32": "f32",
    "float16": "f16",
    "float8_e4m3fnuz": "f8",
    sfnp.int8: "i8",
    sfnp.float32: "f32",
    sfnp.float16: "fp16",
    sfnp.bfloat16: "bf16",
    sfnp.float8_e4m3fnuz: "f8",
}


def get_url_map(filenames: list[str], bucket: str):
    file_map = {}
    for filename in filenames:
        file_map[filename] = f"{bucket}{filename}"
    return file_map


def needs_update(ctx, current_version: str):
    stamp = ctx.allocate_file("version.txt")
    stamp_path = stamp.get_fs_path()
    if os.path.exists(stamp_path):
        with open(stamp_path, "r") as s:
            ver = s.read()
        if ver != current_version:
            return True
    else:
        with open(stamp_path, "w") as s:
            s.write(current_version)
        return True
    return False


# TODO: unify needs_file with needs_file_url
def needs_file(filename, ctx, namespace=FileNamespace.GEN):
    out_file = ctx.allocate_file(filename, namespace=namespace).get_fs_path()
    if os.path.exists(out_file):
        needed = False
    else:
        filekey = os.path.join(ctx.path, filename)
        ctx.executor.all[filekey] = None
        needed = True
    return needed


def needs_file_url(filename, ctx, url=None, namespace=FileNamespace.GEN):
    out_file = ctx.allocate_file(filename, namespace=namespace).get_fs_path()
    needed = True
    if os.path.exists(out_file):
        if url:
            needed = False
            # needed = not is_valid_size(out_file, url)
        if not needed:
            return False
    filekey = os.path.join(ctx.path, filename)
    ctx.executor.all[filekey] = None
    return True


def needs_compile(filename, target, ctx):
    vmfb_name = f"{filename}_{target}.vmfb"
    namespace = FileNamespace.BIN
    return needs_file(vmfb_name, ctx, namespace=namespace)


def get_cached_vmfb(filename, target, ctx):
    vmfb_name = f"{filename}_{target}.vmfb"
    return ctx.file(vmfb_name)


def is_valid_size(file_path, url):
    if not url:
        return True
    with urllib.request.urlopen(url) as response:
        content_length = response.getheader("Content-Length")
    local_size = get_file_size(str(file_path))
    if content_length:
        content_length = int(content_length)
        if content_length != local_size:
            return False
    return True


def get_file_size(file_path):
    """Gets the size of a local file in bytes as an integer."""

    file_stats = os.stat(file_path)
    return file_stats.st_size


def fetch_http_check_size(*, name: str, url: str) -> BuildFile:
    context = BuildContext.current()
    output_file = context.allocate_file(name)
    action = FetchHttpWithCheckAction(
        url=url, output_file=output_file, desc=f"Fetch {url}", executor=context.executor
    )
    output_file.deps.add(action)
    return output_file


class FetchHttpWithCheckAction(BuildAction):
    def __init__(self, url: str, output_file: BuildFile, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.output_file = output_file

    def _invoke(self, retries=4):
        path = self.output_file.get_fs_path()
        self.executor.write_status(f"Fetching URL: {self.url} -> {path}")
        try:
            urllib.request.urlretrieve(self.url, str(path))
        except urllib.error.HTTPError as e:
            if retries > 0:
                retries -= 1
                self._invoke(retries=retries)
            else:
                raise IOError(f"Failed to fetch URL '{self.url}': {e}") from None
        local_size = get_file_size(str(path))
        try:
            with urllib.request.urlopen(self.url) as response:
                content_length = response.getheader("Content-Length")
            if content_length:
                content_length = int(content_length)
                if content_length != local_size:
                    raise IOError(
                        f"Size of downloaded artifact does not match content-length header! {content_length} != {local_size}"
                    )
        except IOError:
            if retries > 0:
                retries -= 1
                self._invoke(retries=retries)


# Common mapping for program isolation modes
PROG_ISOLATIONS = {
    isolation.name.lower(): isolation for isolation in sf.ProgramIsolation
}
# For backward compatibility
prog_isolations = {
    "none": sf.ProgramIsolation.NONE,
    "per_fiber": sf.ProgramIsolation.PER_FIBER,
    "per_call": sf.ProgramIsolation.PER_CALL,
}


class InferenceExecRequest(sf.Message):
    def __init__(self):
        super().__init__()


class StrobeMessage(sf.Message):
    """Sent to strobe a queue with fake activity (generate a wakeup)."""

    ...


class GenerateService:
    """Base class for shortfin service implementations."""

    def __init__(
        self,
        sysman: SystemManager,
        fibers_per_device: int = 1,
        workers_per_device: int = 1,
    ):
        """Initialize base service attributes."""
        self.sysman = sysman
        self.inference_parameters: dict[str, list[sf.BaseProgramParameters]] = {}
        self.inference_modules: dict[str, list[sf.ProgramModule]] = {}
        self.inference_programs: dict[int, dict[str, sf.Program]] = {}
        self.inference_functions: dict[int, dict[str, sf.ProgramFunction]] = {}
        self.name = None
        self.model_params = None
        self.prog_isolation = None
        self.trace_execution = False
        self.show_progress = False

        # Worker and fiber configuration
        self.workers: list[sf.Worker] = []
        self.workers_per_device = workers_per_device
        self.fibers_per_device = fibers_per_device
        self.validate_fiber_configuration()

    def set_isolation(self, isolation_str: str = "per_call"):
        """Set the program isolation mode from a string.

        Args:
            isolation_str: Program isolation mode string
        """
        self.prog_isolation = prog_isolations[isolation_str]

    def validate_fiber_configuration(self):
        """Validate fiber configuration."""
        if self.fibers_per_device % self.workers_per_device != 0:
            raise ValueError(
                "Currently, fibers_per_device must be divisible by workers_per_device"
            )
        self.fibers_per_worker = int(self.fibers_per_device / self.workers_per_device)

    def load_inference_module(
        self, vmfb_path: Path, component: str = "main", batch_size: int = None
    ):
        """Load an inference module from a VMFB file.

        Args:
            vmfb_path: Path to the VMFB file
            component: Optional component name for organizing modules
        """
        if batch_size:
            bs = batch_size
        else:
            match = re.search(r"_bs(\d+)_", str(vmfb_path))
            if match:
                bs = int(match.group(1))
            else:
                bs = None
        if not hasattr(self, "inference_modules"):
            self.inference_modules = {}
        if bs:
            if not self.inference_modules.get(component):
                self.inference_modules[component] = {}
            if not self.inference_modules[component].get(bs):
                self.inference_modules[component][bs] = []
            self.inference_modules[component][bs].append(
                sf.ProgramModule.load(self.sysman.ls, vmfb_path)
            )
        else:
            if not self.inference_modules.get(component):
                self.inference_modules[component] = []
            self.inference_modules[component].append(
                sf.ProgramModule.load(self.sysman.ls, vmfb_path)
            )

    def load_inference_parameters(
        self,
        *paths: Path,
        parameter_scope: str,
        format: str = "",
        component: str = "main",
    ):
        """Load inference parameters from files.

        Args:
            *paths: Paths to parameter files
            parameter_scope: Parameter scope name
            format: Optional format string
            component: Optional component name for organizing parameters
        """
        p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
        for path in paths:
            logging.info("Loading parameter fiber '%s' from: %s", parameter_scope, path)
            p.load(path, format=format)

        if not hasattr(self, "inference_parameters"):
            self.inference_parameters = {}
        if not self.inference_parameters.get(component):
            self.inference_parameters[component] = []
        self.inference_parameters[component].append(p)

    def initialize_program_modules(self, component: str):
        """Initialize program modules for a component.

        Args:
            component: Component name

        Returns:
            List of program modules
        """
        if component not in self.inference_modules:
            return []

        return [
            sf.ProgramModule.parameter_provider(
                self.sysman.ls, *self.inference_parameters.get(component, [])
            ),
            *self.inference_modules[component],
        ]

    def create_program(
        self,
        modules: list[sf.ProgramModule],
        devices: list[sf.Device],
        isolation: Optional[str] = None,
        trace_execution: Optional[bool] = None,
    ) -> sf.Program:
        """Create a program with the given modules and devices.

        Args:
            modules: List of program modules
            devices: List of devices
            isolation: Program isolation mode (defaults to self.prog_isolation)
            trace_execution: Whether to trace execution (defaults to self.trace_execution)

        Returns:
            Program instance
        """
        if isolation is None:
            isolation = self.prog_isolation

        if trace_execution is None:
            trace_execution = self.trace_execution

        return sf.Program(
            modules=modules,
            devices=devices,
            trace_execution=trace_execution,
            isolation=isolation,
        )

    def create_worker(self, device: sf.Device, index: int) -> sf.Worker:
        """Create a worker for a device.

        Args:
            device: Device to create worker for
            index: Worker index

        Returns:
            Worker instance
        """
        return self.sysman.ls.create_worker(
            f"{self.name}-inference-{device.name}-{index}"
        )

    def start(self):
        """Start the service by loading program modules and launching the batcher."""
        # Override in derived classes
        pass

    def shutdown(self):
        """Shutdown the service."""
        if hasattr(self, "batcher"):
            self.batcher.shutdown()

    def __repr__(self):
        """Default string representation for service instances."""
        modules = [
            f"     {key} : {value}" for key, value in self.inference_modules.items()
        ]
        params = [
            f"     {key} : {value}" for key, value in self.inference_parameters.items()
        ]
        # For python 3.11 since we can't have \ in the f"" expression.
        new_line = "\n"
        return (
            f"ServiceManager("
            f"\n  INFERENCE DEVICES : \n"
            f"     {self.sysman.ls.devices}\n"
            f"\n  MODEL PARAMS : \n"
            f"{self.model_params}"
            f"\n  SERVICE PARAMS : \n"
            f"     fibers per device : {self.fibers_per_device}\n"
            f"     program isolation mode : {self.prog_isolation}\n"
            f"\n  INFERENCE MODULES : \n"
            f"{new_line.join(modules)}\n"
            f"\n  INFERENCE PARAMETERS : \n"
            f"{new_line.join(params)}\n"
            f")"
        )


class BatcherProcess(sf.Process):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches."""

    STROBE_SHORT_DELAY = 0.5
    STROBE_LONG_DELAY = 1.0

    def __init__(self, fiber, name="batcher"):
        super().__init__(fiber=fiber)
        self.batcher_infeed = self.system.create_queue()
        self.strobe_enabled = True
        self.strobes = 0
        self.pending_requests = set()
        self.logger = logging.getLogger("batcher")

    def shutdown(self):
        """Shutdown the batcher process."""
        self.batcher_infeed.close()

    def submit(self, request):
        """Submit a request to the batcher."""
        self.batcher_infeed.write_nodelay(request)

    async def _background_strober(self):
        """Background strober task that monitors pending requests."""
        while not self.batcher_infeed.closed:
            await asyncio.sleep(
                self.STROBE_SHORT_DELAY
                if len(self.pending_requests) > 0
                else self.STROBE_LONG_DELAY
            )
            if self.strobe_enabled:
                self.submit(StrobeMessage())

    def custom_message(self, msg):
        self.logger.error("Illegal message received by batcher: %r", msg)
        exit(1)

    async def run(self):
        """Main run loop for the batcher process."""
        strober_task = asyncio.create_task(self._background_strober())
        reader = self.batcher_infeed.reader()
        while item := await reader():
            self.strobe_enabled = False
            if isinstance(item, InferenceExecRequest):
                self.handle_inference_request(item)
            elif isinstance(item, StrobeMessage):
                self.strobes += 1
            else:
                self.custom_message(item)
            await self.process_batches()

            self.strobe_enabled = True
        await strober_task

    def handle_inference_request(self, request):
        """Handle an inference request. To be implemented by subclasses."""
        pass

    async def process_batches(self):
        """Process batches of requests. To be implemented by subclasses."""
        pass

    def sort_batches(self):
        """Files pending requests into sorted batches suitable for program invocations.

        This is a common implementation used by SDXLBatcherProcess and FluxBatcherProcess.
        """
        if not hasattr(self, "pending_requests"):
            return {}

        reqs = self.pending_requests
        next_key = 0
        batches = {}
        for req in reqs:
            is_sorted = False
            req_metas = [req.phases[phase]["metadata"] for phase in req.phases.keys()]

            for idx_key, data in batches.items():
                if not isinstance(data, dict):
                    logger.error(
                        "Expected to find a dictionary containing a list of requests and their shared metadatas."
                    )
                if len(batches[idx_key]["reqs"]) >= self.ideal_batch_size:
                    # Batch is full
                    next_key = idx_key + 1
                    continue
                elif data["meta"] == req_metas:
                    batches[idx_key]["reqs"].extend([req])
                    is_sorted = True
                    break
                else:
                    next_key = idx_key + 1
            if not is_sorted:
                batches[next_key] = {
                    "reqs": [req],
                    "meta": req_metas,
                }
        return batches
