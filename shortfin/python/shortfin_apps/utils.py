from iree.build.executor import FileNamespace, BuildAction, BuildContext, BuildFile
import os
import urllib
import logging
import asyncio
from pathlib import Path

import shortfin.array as sfnp
import shortfin as sf
from shortfin_apps.flux.components.manager import SystemManager


dtype_to_filetag = {
    "bfloat16": "bf16",
    "float32": "f32",
    "float16": "f16",
    sfnp.int8: "i8",
    sfnp.float32: "f32",
    sfnp.float16: "fp16",
    sfnp.bfloat16: "bf16",
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


class ServiceBase:
    """Base class for shortfin service implementations."""

    def __init__(self, sysman: SystemManager):
        """Initialize base service attributes."""
        self.sysman = sysman
        self.inference_parameters: dict[str, list[sf.BaseProgramParameters]] = {}
        self.inference_modules: dict[str, list[sf.ProgramModule]] = {}

    def load_inference_module(self, vmfb_path: Path, component: str = "main", batch_size: int = None):
        """Load an inference module from a VMFB file.

        Args:
            vmfb_path: Path to the VMFB file
            component: Optional component name for organizing modules
        """
        if not hasattr(self, "inference_modules"):
            self.inference_modules = {}
        if not self.inference_modules.get(component):
            self.inference_modules[component] = {}

        if batch_size:
            bs = batch_size
        else:
            match = re.search(r"_bs(\d+)_", str(vmfb_path))
            if match:
                bs = int(match.group(1))
            else:
                raise ValueError(
                    "Batch size not found in filename or provided to load function."
                )
        if not self.inference_modules[component].get(bs):
            self.inference_modules[component][bs] = []
        self.inference_modules[component][bs].append(
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


class BatcherProcessBase(sf.Process):
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
                logger.error("Illegal message received by batcher: %r", item)
                exit(1)

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
