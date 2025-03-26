from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class DeviceSettings:
    compile_flags: Tuple[str]
    server_flags: Tuple[str]
    name: str = ""  # Name of the device setting for reference


# Base device settings (non-TP versions)
CPU = DeviceSettings(
    name="cpu",
    compile_flags=(
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ),
    server_flags=("--device=local-task",),
)

GFX942 = DeviceSettings(
    name="gfx942",
    compile_flags=(
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx942",
    ),
    server_flags=("--device=hip",),
)

GFX90A = DeviceSettings(
    name="gfx90a",
    compile_flags=(
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx90a",
    ),
    server_flags=("--device=hip",),
)

GFX1100 = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-backends=rocm",
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


def create_tp_settings_cpu(tp_count: int) -> DeviceSettings:
    """
    Create tensor parallelism settings for CPU devices.

    Args:
        tp_count: Number of devices to use for tensor parallelism

    Returns:
        DeviceSettings with appropriate flags for CPU tensor parallelism
    """

    compile_flags = []

    # compile
    compile_flags.extend(
        [f"--iree-hal-target-device=llvm-cpu[{i}]" for i in range(tp_count)]
    )

    # serve
    server_flags = [f"--device=local-task"]
    server_flags.extend(["--device_ids"] + [str(i) for i in range(tp_count)])

    return DeviceSettings(
        name=f"cpu_tp{tp_count}",
        compile_flags=tuple(compile_flags),
        server_flags=tuple(server_flags),
    )


def create_tp_settings_hip(gfx_level: str, tp_count: int) -> DeviceSettings:
    """
    Create tensor parallelism settings for HIP devices.

    Args:
        gfx_level: GFX level name (e.g., "gfx942")
        tp_count: Number of devices to use for tensor parallelism

    Returns:
        DeviceSettings with appropriate flags for HIP tensor parallelism
    """
    if not gfx_level.startswith("gfx"):
        raise ValueError(
            f"Device {gfx_level} is not a valid HIP device. HIP devices should start with 'gfx'."
        )

    # compile
    compile_flags = [f"--iree-hip-target={gfx_level}"]
    compile_flags.extend(
        [f"--iree-hal-target-device=hip[{i}]" for i in range(tp_count)]
    )

    # serve
    server_flags = [f"--device=hip"]
    server_flags.extend(["--device_ids"] + [str(i) for i in range(tp_count)])

    return DeviceSettings(
        name=f"{gfx_level}_tp{tp_count}",
        compile_flags=tuple(compile_flags),
        server_flags=tuple(server_flags),
    )


def get_device_settings_by_name(device_name):
    """
    Get device settings by name.

    Device names can be one of:
    - Base device names from the table (e.g., "gfx942", "gfx90a", "cpu")
    - Tensor parallel variants in the format "{base_device}_tp{n}" (e.g., "gfx942_tp2")
      where n is the number of devices to use (2, 4, 8, etc.)

    The function will dynamically generate appropriate settings for tensor parallelism
    if they aren't already in the table.
    """
    device_name = device_name.lower()

    # Check if it's in the predefined table
    if device_name in table:
        return table[device_name]

    # Check if it's a tensor parallel variant
    import re

    # Handle both GPU and CPU tensor parallelism formats
    # Format: {device_type}_tp{count} - e.g., gfx942_tp2, cpu_tp4
    tp_match = re.match(r"^([a-z0-9]+)_tp(\d+)$", device_name)

    if tp_match:
        base_device = tp_match.group(1)
        tp_count = int(tp_match.group(2))

        # Verify base device exists
        if base_device not in table:
            raise ValueError(f"Base device '{base_device}' not recognized")

        # Dynamically create tensor parallelism settings
        if base_device.startswith("gfx"):
            tp_settings = create_tp_settings_hip(base_device, tp_count)
        elif base_device in ["cpu", "host", "hostcpu", "local-task"]:
            tp_settings = create_tp_settings_cpu(tp_count)

        # Cache the settings for future use
        table[device_name] = tp_settings

        return tp_settings

    # Get unique base device names
    base_devices = set(table.keys())
    raise ValueError(
        f"Device name '{device_name}' is not recognized. Supported device names: {list(table.keys())} "
        f"or any base device with tensor parallelism in the format 'device_tpN' where device is "
        f"one of {list(base_devices)} and N is the number of devices (e.g., cpu_tp2, gfx942_tp4)."
    )


# Comprehensive test function to verify dynamic TP flag generation
def _test_tp_generation():
    """Test tensor parallelism settings generation for both CPU and GPU devices."""

    # Test cached settings
    for device, size in [("gfx942", 2), ("gfx942", 4), ("cpu", 2), ("cpu", 4)]:
        print(f"Testing device: {device} size: {size}")
        key = f"{device}_tp{size}"
        settings = get_device_settings_by_name(key)
        print(f"got settings: {settings}")
        # Verify it has the correct number of device targets
        assert len([f for f in settings.compile_flags if "target-device" in f]) == size
        # Verify it has the correct number of device IDs
        assert len([f for f in settings.server_flags if f.isdigit()]) == size

    # Test dynamic generation for new sizes
    for device, sizes in [("gfx942", [3, 8]), ("gfx90a", [2, 8]), ("cpu", [3, 8])]:
        for size in sizes:
            key = f"{device}_tp{size}"
            # Skip if already in table
            if key in table:
                del table[key]  # Remove it to force regeneration

            settings = get_device_settings_by_name(key)
            # Verify it has the correct number of device targets
            assert (
                len([f for f in settings.compile_flags if "target-device" in f]) == size
            )
            # Verify it has the correct number of device IDs
            assert len([f for f in settings.server_flags if f.isdigit()]) == size
            # Just check the device target count
            # Verify correct flags for device type
            if device.startswith("gfx"):
                assert "--device=hip" in settings.server_flags
                assert f"--iree-hip-target={device}" in settings.compile_flags
            else:
                assert "--device=local-task" in settings.server_flags
                assert "--iree-hal-target-device=llvm-cpu[0]" in settings.compile_flags

    # Verify error for non-existent device
    try:
        get_device_settings_by_name("nonexistent_device")
        assert False, "Should have raised ValueError for non-existent device"
    except ValueError:
        pass

    # Verify error for non-existent base device with TP
    try:
        get_device_settings_by_name("nonexistent_tp4")
        assert (
            False
        ), "Should have raised ValueError for non-existent base device with TP"
    except ValueError:
        pass

    # Verify we can get cached TP settings
    key = "gfx942_tp8"
    assert key in table, "TP settings should be cached after generation"

    print("All dynamic TP generation tests passed!")


# Run tests when module is executed directly
if __name__ == "__main__":
    _test_tp_generation()
