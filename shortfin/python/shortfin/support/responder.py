# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from shortfin.support.status_tracker import AbstractStatusTracker


class AbstractResponder:
    """Interface for a responder to"""

    def __init__(self):
        pass

    def start_response(self):
        pass

    def ensure_response(self):
        pass

    def send_response(self, response):
        pass

    def stream_start(self, **kwargs):
        pass

    def stream_part(self, content: bytes | None):
        pass

    def get_status_tracker(self):
        return AbstractStatusTracker()

    def is_disconnected(self):
        return False
