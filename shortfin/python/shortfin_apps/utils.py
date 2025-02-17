from iree.build.executor import FileNamespace, BuildAction, BuildContext, BuildFile
import os
import urllib

import shortfin.array as sfnp


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
