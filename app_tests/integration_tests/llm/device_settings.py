from typing import Tuple
from dataclasses import dataclass


@dataclass
class DeviceSettings:
    compile_flags: Tuple[str]
    server_flags: Tuple[str]


CPU = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ),
    server_flags=("--device=local-task",),
)

GFX942 = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-device=hip",
        "--iree-hip-target=gfx942",
    ),
    server_flags=("--device=hip",),
)

GFX90A = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-device=hip",
        "--iree-hip-target=gfx90a",
    ),
    server_flags=("--device=hip",),
)

GFX1100 = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-device=hip",
        "--iree-hip-target=gfx1100",
    ),
    server_flags=("--device=hip",),
)

table = {
    "gfx942": GFX942,
    "gfx90a": GFX90A,
    "gfx1100": GFX1100,
    "host": CPU,
    "hostcpu": CPU,
    "local-task": CPU,
    "cpu": CPU,
}


def get_device_settings_by_name(device_name):
    """
    Get device settings by name.

    See table below for options:
    """

    if device_name.lower() in table:
        return table[device_name]

    raise ValueError(
        f"os.environ['SHORTFIN_INTEGRATION_TEST_DEVICE']=={device_name} but is not recognized. Supported device names: {list(table.keys())}"
    )
