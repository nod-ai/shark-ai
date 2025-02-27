# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin_apps.utils import SystemManager


class SDXLSystemManager(SystemManager):
    def __init__(self, device="local-task", device_ids=None, async_allocs=True):
        super().__init__(
            device=device,
            device_ids=device_ids,
            async_allocs=async_allocs,
            logger_name="shortfin-sd.manager",
            shutdown_system=True,
        )
