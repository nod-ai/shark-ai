# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class AbstractResponder:
    """Interface for a responder to"""

    def __init__(self):
        pass

    def ensure_response(self):
        pass

    def send_response(self, response):
        pass

    def start_response(self, **kwargs):
        pass

    def stream_part(self, content: bytes | None):
        pass
