# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...utils import (
    SystemManager,
)
from .fiber_pool import FiberPool
from shortfin import Fiber


class StreamManager:
    def __init__(
        self,
        sysman: SystemManager,
        pool_initialization_size: int,
    ):

        self.__sysman = sysman
        self.__disaggregate = self.__sysman.disaggregate
        self.__devices = self.__sysman.ls.devices
        self.__pool_init_size = pool_initialization_size
        self.__fiber_pool = self.__construct_fiber_pool()
        self.__create_fiber = self.__sysman.ls.create_fiber
        self.__create_worker = self.__sysman.ls.create_worker
        self.__stream_idx = -1

    def __construct_fiber_pool(self):
        return FiberPool(
            sysman=self.__sysman,
            init_size=self.__pool_init_size,
            resizable=True,
            name="stream_managed_fiber_pool",
        )

    def fiber_pool(self):
        return self.__fiber_pool

    def num_open_streams(self) -> int:
        if not self.__disaggregate:
            return 1
        return len(self.__sysman.ls.devices)

    def construct_main_fibers(self) -> tuple[Fiber]:
        # TODO(vinayakdsci): This code path assumes right now that we are only
        # going to create two streams for disaggregation. As disaggregation scales,
        # this assumption will need to be revisited.
        tasks = [
            "prefill-batcher",
            "decode-batcher",
            "prefill-executor",
            "decode-executor",
            "main",
        ]
        if not self.__disaggregate:
            return tuple(
                [
                    self.__create_fiber(
                        self.__create_worker(f"default-{task}-worker-0")
                    )
                    for task in tasks
                ]
            )

        return tuple(
            [
                self.__create_fiber(
                    self.__create_worker(
                        f"default-disaggregated-stream-{task}-worker-0"
                    ),
                    devices=[self.__devices[idx % len(self.__devices)]],
                )
                for idx, task in enumerate(tasks)
            ]
        )

    def get_stream(self):
        if not self.__disaggregate:
            return (
                0,
                self.__sysman.ls.devices,
            )

        self.__stream_idx += 1
        return (
            self.__stream_idx,
            [
                self.__sysman.ls.devices[
                    self.__stream_idx % len(self.__sysman.ls.devices)
                ]
            ],
        )
