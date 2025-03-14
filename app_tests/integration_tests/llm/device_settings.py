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

# Base mapping of device names to settings
table = {
    "gfx942": GFX942,
    "gfx90a": GFX90A,
    "host": CPU,
    "hostcpu": CPU,
    "local-task": CPU,
    "cpu": CPU,
}


# Helper function to generate tensor parallelism settings
def create_tp_settings(base_device: str, tp_count: int) -> DeviceSettings:
    """
    Create tensor parallelism settings for a given base device and count.

    Args:
        base_device: Base device name (e.g., "cpu", "gfx942")
        tp_count: Number of devices to use for tensor parallelism

    Returns:
        DeviceSettings with appropriate flags for tensor parallelism
    """
    # Get base device settings
    if base_device in ["cpu", "host", "hostcpu", "local-task"]:
        # CPU tensor parallelism - just use the HAL device targets for multi-device
        compile_flags = [
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
        ]
        for i in range(tp_count):
            compile_flags.append(f"--iree-hal-target-device=local-task[{i}]")

        server_flags = ["--device=local-task", "--device_ids"]
        for i in range(tp_count):
            server_flags.append(f"{i}")

        return DeviceSettings(
            name=f"cpu_tp{tp_count}",
            compile_flags=tuple(compile_flags),
            server_flags=tuple(server_flags),
        )
    else:
        # GPU tensor parallelism - just use the HAL device targets for multi-device
        compile_flags = [
            "--iree-hal-target-backends=rocm",
            f"--iree-hip-target={base_device}",
        ]
        for i in range(tp_count):
            compile_flags.append(f"--iree-hal-target-device=hip[{i}]")

        server_flags = ["--device=hip", "--device_ids"]
        for i in range(tp_count):
            server_flags.append(f"{i}")

        return DeviceSettings(
            name=f"{base_device}_tp{tp_count}",
            compile_flags=tuple(compile_flags),
            server_flags=tuple(server_flags),
        )


# Add a few common TP settings to the table for quick access
table["cpu_tp2"] = create_tp_settings("cpu", 2)
table["cpu_tp4"] = create_tp_settings("cpu", 4)
table["gfx942_tp2"] = create_tp_settings("gfx942", 2)
table["gfx942_tp4"] = create_tp_settings("gfx942", 4)


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
        tp_settings = create_tp_settings(base_device, tp_count)

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


# Comprehensive test function to verify dynamic TP generation
def _test_tp_generation():
    """Test tensor parallelism settings generation for both CPU and GPU devices."""

    # Test cached settings
    for device, size in [("gfx942", 2), ("gfx942", 4), ("cpu", 2), ("cpu", 4)]:
        key = f"{device}_tp{size}"
        settings = get_device_settings_by_name(key)
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
                assert "--iree-hal-target-backends=llvm-cpu" in settings.compile_flags

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
