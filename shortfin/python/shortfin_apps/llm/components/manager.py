# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin_apps.utils import SystemManager


class LlmSystemManager(SystemManager):
    def __init__(
        self,
        device="local-task",
        device_ids=None,
        async_allocs=True,
        amdgpu_allocators=None,
        amdgpu_allow_device_reuse=False,
    ):
        super().__init__(
            device=device,
            device_ids=device_ids,
            async_allocs=async_allocs,
            amdgpu_allocators=amdgpu_allocators,
            amdgpu_allow_device_reuse=amdgpu_allow_device_reuse,
            logger_name=__name__,
            shutdown_system=False,
        )
