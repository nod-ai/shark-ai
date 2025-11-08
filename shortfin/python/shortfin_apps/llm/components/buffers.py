import logging

from typing import List, Optional, Sequence, Tuple

import shortfin as sf
import shortfin.array as sfnp

from .device_array_cache import Allocation


logger = logging.getLogger(__name__)


def create_argument_buffers(
    buffers: List[Allocation],
    data: Sequence[List[int | float] | List[List[int | float]]],
    defaults: List[Optional[int | float]],
) -> List[Allocation]:
    """Create argument buffers to submit to VMFB for prefill or decode.

    `buffers`, `data`, and `defaults` are parallel lists.

    Args:
        buffers (List[Allocation]): Buffers to passed to VMFB.
        data (Sequence[List[int | float] | List[List[int | float]]]): Data to fill the buffers with.
        defaults (List[Optional[int  |  float]]): Defaults to fill the buffers with.

    Returns:
        List[sfnp.device_array]: A list of device arrays corresponding to the input buffers.
    """
    assert len(buffers) == len(data), "`buffers` and `data` must be parallel lists"
    if defaults is not None:
        assert len(buffers) == len(
            defaults
        ), "`buffers` and `defaults` must be parallel lists"

    args = []
    for index, buffer in enumerate(buffers):
        buffer_data = data[index]
        default = defaults[index]
        with buffer.host.map(discard=True) as host_buffer:
            if default is not None:
                host_buffer.fill(default)
            host_buffer.items = buffer_data
            # Log the final buffer contents for debugging
            logger.info(
                f"Buffer {index}: shape={buffer.shape}, data={list(host_buffer.items)[:min(20, len(host_buffer.items))]}"
            )

        args.append(buffer)

    [buffer.transfer_to_device() for buffer in buffers]
    return args


async def copy_buffers_to_host(
    buffers: Tuple[Optional[sfnp.device_array]],
    device: sf.ScopedDevice,
) -> List[sfnp.device_array]:
    """Copy device buffers to host buffers.

    This function takes a list of device arrays and copies each one to a host buffer.
    If a buffer is `None`, it appends `None` to the new buffers list.

    Args:
        buffers (List[Optional[sfnp.device_array]]): List of device arrays to copy to host.

    Returns:
        List[sfnp.device_array]: A list of host buffers corresponding to the input device buffers.
    """
    new_buffers = []
    for buffer in buffers:
        if buffer is None:
            new_buffers.append(None)
            continue

        host_buffer = buffer.for_transfer()
        host_buffer.copy_from(buffer)
        new_buffers.append(host_buffer)

    await device
    return new_buffers
