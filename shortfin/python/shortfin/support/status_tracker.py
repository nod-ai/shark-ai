# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio


class AbstractStatusTracker:
    """Interface for tracking request or connection status, such as disconnection.
    Extend this class to add more status checks in the future.
    """

    def __init__(self):
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            print("Warning: No running loop found")

    def is_disconnected(self) -> bool:
        """Returns True if the connection/request is considered disconnected."""
        raise NotImplementedError()
