# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...utils import (
    SystemManager,
    GenerateService,
    LLM_DISAGGREGATED_DECODE_DEVICE_IDX,
    LLM_DISAGGREGATED_PREFILL_DEVICE_IDX,
)
from .fiber_pool import FiberPool
from shortfin import Fiber

LLM_NONDISAGGREGATED_MODULE_IDX = 0


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

    def __construct_fiber_pool(self):
        return FiberPool(
            sysman=self.__sysman,
            init_size=self.__pool_init_size,
            resizable=True,
            name="stream_managed_fiber_pool",
        )

    def fiber_pool(self):
        return self.__fiber_pool

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

    def __init_prog_modules(self, service):
        component_modules = service.initialize_program_modules("main")
        if not self.__disaggregate:
            service.inference_program = [
                service.create_program(
                    modules=component_modules, devices=service.sysman.ls.devices
                )
            ]
        else:
            service.inference_program = [
                service.create_program(
                    modules=component_modules, devices=[self.__devices[idx]]
                )
                for idx in range(len(self.__devices))
            ]

        service.prefill_functions = {}
        service.decode_functions = {}

        prefill_device_idx = (
            LLM_NONDISAGGREGATED_MODULE_IDX
            if not self.__disaggregate
            else LLM_DISAGGREGATED_PREFILL_DEVICE_IDX
        )
        decode_device_idx = (
            LLM_NONDISAGGREGATED_MODULE_IDX
            if not self.__disaggregate
            else LLM_DISAGGREGATED_DECODE_DEVICE_IDX
        )

        for bs in service.model_params.prefill_batch_sizes:
            service.prefill_functions[bs] = service.inference_program[
                prefill_device_idx
            ][f"{service.model_params.module_name}.prefill_bs{bs}"]
        # Resolve decode entrypoints.
        service.decode_functions = {}
        for bs in service.model_params.decode_batch_sizes:
            service.decode_functions[bs] = service.inference_program[decode_device_idx][
                f"{service.model_params.module_name}.decode_bs{bs}"
            ]

    def load_program_modules(self, service: GenerateService):
        self.__init_prog_modules(service)
