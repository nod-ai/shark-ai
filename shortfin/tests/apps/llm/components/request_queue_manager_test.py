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
    assert queue_manager.add_to_queue(get_decode_configs(4))

    # Try to add beyond max, when `current_queue_size` < `max_queue_size`
    assert not queue_manager.add_to_queue(get_decode_configs(3))

    # Add more to queue
    assert queue_manager.add_to_queue(get_decode_configs(2))

    # Try to add beyond max
    assert not queue_manager.add_to_queue(get_decode_configs(2))

    # Remove from queue
    queue_manager.remove_from_queue(get_decode_configs(3))
