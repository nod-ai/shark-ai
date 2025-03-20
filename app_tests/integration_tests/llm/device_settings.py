from typing import Tuple
from dataclasses import dataclass


@dataclass
class DeviceSettings:
    compile_flags: Tuple[str]
    server_flags: Tuple[str]


CPU = DeviceSettings(
    compile_flags=(
        "-iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ),
    server_flags=("--device=local-task",),
)

GFX942 = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx942",
    ),
    server_flags=("--device=hip",),
)

GFX942_TP4 = DeviceSettings(
    compile_flags=(
        "--iree-hip-target=gfx942",
        "--iree-hal-target-device=hip[0]",
        "--iree-hal-target-device=hip[1]",
        "--iree-hal-target-device=hip[2]",
        "--iree-hal-target-device=hip[3]",
    ),
    server_flags=(
        "--device=hip",
        "--device_ids",
        "0",
        "1",
        "2",
        "3",
    ),
    # server_flags=(
    #     "--device=hip",
    #     "--device_ids",
    #     "0",
    #     "0",
    #     "0",
    #     "0",
    # ),  # temporarily testing on all 4 device actually being the same device
)

GFX90A = DeviceSettings(
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


def get_device_settings_by_name(device_name):
    """
    Get device settings by name.

    Device names can be one of:
    - Base device names from the table (e.g., "gfx942", "gfx90a", "cpu")
    - Tensor parallel variants in the format "{base_device}_tp{n}" (e.g., "gfx942_tp2")
      where n is the number of devices to use (2, 4, 8, etc.)
    """
    device_name = device_name.lower()

    # Check if it's in the predefined table
    if device_name in table:
        return table[device_name]

    # Check if it's a tensor parallel variant
    import re

    tp_match = re.match(r"^(gfx\d+)_tp(\d+)$", device_name)

    if tp_match:
        base_device = tp_match.group(1)
        tp_count = int(tp_match.group(2))

        # Verify base device exists
        if base_device not in table:
            raise ValueError(f"Base device '{base_device}' not recognized")

        # Get base device settings
        base_settings = table[base_device]

        # Generate compile flags
        compile_flags = [f"--iree-hip-target={base_device}"]
        for i in range(tp_count):
            compile_flags.append(f"--iree-hal-target-device=hip[{i}]")

        # Generate server flags
        server_flags = ["--device=hip", "--device_ids"]
        for i in range(tp_count):
            server_flags.append(f"{i}")

        return DeviceSettings(
            compile_flags=tuple(compile_flags), server_flags=tuple(server_flags)
        )

    raise ValueError(
        f"os.environ['SHORTFIN_INTEGRATION_TEST_DEVICE']=={device_name} but is not recognized. Supported device names: {list(table.keys())} or {base_device}_tp{n} format"
    )


# Simple test function to verify dynamic TP generation
def _test_tp_generation():
    # Test gfx942_tp2
    tp2_settings = get_device_settings_by_name("gfx942_tp2")
    assert len([f for f in tp2_settings.compile_flags if "target-device" in f]) == 2
    assert len([f for f in tp2_settings.server_flags if f.isdigit()]) == 2

    # Test gfx942_tp4
    tp4_settings = get_device_settings_by_name("gfx942_tp4")
    assert len([f for f in tp4_settings.compile_flags if "target-device" in f]) == 4
    assert len([f for f in tp4_settings.server_flags if f.isdigit()]) == 4

    # Test gfx90a_tp8
    tp8_settings = get_device_settings_by_name("gfx90a_tp8")
    assert len([f for f in tp8_settings.compile_flags if "target-device" in f]) == 8
    assert len([f for f in tp8_settings.server_flags if f.isdigit()]) == 8

    print("All dynamic TP generation tests passed!")


# Uncomment to run tests
# if __name__ == "__main__":
#     _test_tp_generation()
