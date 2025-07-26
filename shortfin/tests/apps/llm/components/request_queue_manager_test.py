# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin_apps.llm.components.token_selection_strategy.config import DecodeConfig
from shortfin_apps.llm.components.request_queue_manager import RequestQueueManager


def get_decode_configs(beam_count):
    return [DecodeConfig(eos_token_id=0, num_beams=beam_count)]


def test_request_queue_manager():
    queue_manager = RequestQueueManager(6)

    # Add to queue
    id0 = queue_manager.add_to_queue(get_decode_configs(4))
    assert id0 is not None

    # Try to add beyond max, when `current_queue_size` < `max_queue_size`
    id1 = queue_manager.add_to_queue(get_decode_configs(3))
    assert id1 is None

    # Add more to queue
    id2 = queue_manager.add_to_queue(get_decode_configs(2))
    assert id2 is not None

    # Try to add beyond max
    id3 = queue_manager.add_to_queue(get_decode_configs(2))
    assert id3 is None

    # Remove from queue
    queue_manager.remove_from_queue(id2)

    tasks = queue_manager.current_tasks()
    assert len(tasks) == 1
    assert id0 in tasks
