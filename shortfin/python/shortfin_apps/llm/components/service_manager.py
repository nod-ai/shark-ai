# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import multiprocessing

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import JoinableQueue
import os
from pathlib import Path
from typing import List, Tuple

from .config_struct import ModelParams, ServerParams
from .token_selection_strategy import DecodeConfig
from .generate import ClientGenerateBatchProcess
from .io_struct import GenerateReqInput
from .manager import LlmSystemManager
from .service import LlmGenerateService
from .service_debug_dumper import SERVICE_DEBUG_DUMPER
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class LlmServiceEnvironment:
    """
    Environment  for LLM services, responsible for creating and managing
    instances of LlmGenerateService.

    This class, which also holds the shortfin system, is a singleton per
    process. For single-process usage, create one of these at the server
    (lifecycle) level. For multi-process usage, create one of these in each
    subprocess.
    """

    @staticmethod
    def get_server_params(args) -> ServerParams:
        server_params = ServerParams.load(
            args.server_config if hasattr(args, "server_config") else None
        )
        server_params.update_from_args(args)
        return server_params

    @staticmethod
    def get_model_params(model_config: Path) -> ModelParams:
        model_params = ModelParams.load_json(model_config)
        return model_params

    def __init__(self, args, create_multiple_services: bool):
        # Load server configuration with priority:
        # command line > config file > defaults
        model_params = LlmServiceEnvironment.get_model_params(args.model_config)
        server_params = LlmServiceEnvironment.get_server_params(args)
        self.num_instances = server_params.instances if create_multiple_services else 1
        decode_bs = model_params.decode_batch_sizes[-1]
        if server_params.decode_config is None:
            decode_config = DecodeConfig(
                args.num_beams,
                args.token_selection_strategy,
                logits_normalization=model_params.logits_normalization,
                max_decode_batch_size=decode_bs,
            )
            server_params.decode_config = decode_config

        # Setup system (configure devices, etc).
        sysman = LlmSystemManager(
            device=args.device,
            device_ids=server_params.device_ids,
            async_allocs=server_params.amdgpu_async_allocations,
            amdgpu_allocators=server_params.amdgpu_allocators,
            amdgpu_allow_device_reuse=server_params.amdgpu_allow_device_reuse,
        )

        # Setup each service we are hosting.
        eos_token = LlmServiceProcess.get_eos_from_tokenizer_config(
            args.tokenizer_config_json
        )
        tokenizer = Tokenizer.from_tokenizer_json_file(
            args.tokenizer_json, eos_token=eos_token
        )
        print(f"Creating {self.num_instances} instances")
        # Create a list of LLM instances
        services: List[LlmGenerateService] = {}
        for i in range(self.num_instances):
            service = LlmGenerateService(
                name=f"instance{i}",
                instance_num=i,
                sysman=sysman,
                tokenizer=tokenizer,
                model_params=model_params,
                server_params=server_params,
                program_isolation=server_params.program_isolation,
            )
            service.load_inference_module(args.vmfb)
            service.load_inference_parameters(*args.parameters, parameter_scope="model")
            services[i] = service

        self.sysman = sysman
        self.services = services

    def start(self):
        self.sysman.start()
        for service_name, service in self.services.items():
            print(f"Initializing service '{service_name}'")
            service.start()

    def shutdown(self):
        for service_name, service in self.services.items():
            print(f"Shutting down service '{service_name}'")
            service.shutdown()
        self.sysman.shutdown()


class LlmServiceProcess(multiprocessing.Process):
    @staticmethod
    def get_eos_from_tokenizer_config(json_path):
        import json

        with open(json_path, "rt") as f:
            json_text = f.read()
        config = json.loads(json_text)
        return config["eos_token"]

    def __init__(
        self, args, request_queue: JoinableQueue, response_queue: JoinableQueue
    ):
        super().__init__()
        self.args = args
        self.request_queue: JoinableQueue = request_queue
        self.response_queue: JoinableQueue = response_queue
        self.service_environment = None
        self.service = None  # Single instance in this process

    def run(self):
        print(f"Starting LLM service process {self.name}")
        self.service_environment = LlmServiceEnvironment(
            self.args, create_multiple_services=False
        )
        self.service = self.service_environment.services[
            0
        ]  # Single instance in this process
        self.service_environment.start()
        print(f"LlmServiceProcess {self.name} ready")

        while True:
            request_packet: Tuple[int, GenerateReqInput] = self.request_queue.get()
            self.request_queue.task_done()
            request_counter = request_packet[0]
            request = request_packet[1]
            # print(f"LlmServiceProcess {self.name} received request packet: {request_counter}")

            if request is None:
                # Shutdown signal
                self.response_queue.put((0, None))
                break

            if not self.service.add_to_queue():
                logger.warning(
                    f"Queue full, dropping request {request_counter}: {request}"
                )
                continue

            def response_handler(response, response_counter=request_counter):
                # print(f"LlmServiceProcess {self.name} sending response packet {response_counter}")
                self.response_queue.put((response_counter, response))
                self.service.remove_from_queue()

            ClientGenerateBatchProcess(self.service, request, response_handler).launch()
            # Don't wait for it to finish! response_handler should take care of
            # notifying when the response is ready. Waiting here increases
            # latency

        print(f"Shutting down LLM service process {self.name}")
        self.service_environment.shutdown()


class LlmServiceManager:
    """
    Base class for managing LLM service instances.
    """

    def __init__(self, args):
        self.args = args
        self.cur_instance_num = 0
        self.request_counter = 0
        self.num_instances = 0

    def start(self):
        """
        Start all resources contained in this service manager.
        """
        pass

    async def send_request(self, gen_req: GenerateReqInput, response_handler: callable):
        """
        Send a request to an LLM service instance. The response is sent
        to the given callable and takes the form of a single byte array or a
        list of byte arrays.
        """
        pass

    def shutdown(self):
        """
        Shutdown all resources contained in this service manager.
        """
        pass

    def get_next_instance_num(self) -> int:
        """
        Get the next service instance number in a round-robin fashion.
        """
        self.cur_instance_num += 1
        if self.cur_instance_num >= self.num_instances:
            self.cur_instance_num = 0
        return self.cur_instance_num


class LlmSingleProcessServiceManager(LlmServiceManager):
    """
    Manages all service instances in a single process.
    """

    def __init__(self, args):
        super().__init__(args)
        self.service_environment = LlmServiceEnvironment(
            args, create_multiple_services=True
        )
        self.num_instances = self.service_environment.num_instances

    def start(self):
        self.service_environment.start()

    async def send_request(self, gen_req: GenerateReqInput, response_handler: callable):
        self.request_counter += 1
        # instance_num = self.get_next_instance_num()
        # Fill up the queue of the current instance first, so that the batcher
        # can dispatch a batch ASAP
        instance_num = self.cur_instance_num
        while not self.service_environment.services[instance_num].add_to_queue():
            instance_num = self.get_next_instance_num()
            await asyncio.sleep(0.001)

        service = self.service_environment.services[instance_num]

        def response_handler_wrapper(response, service=service):
            response_handler(response)
            service.remove_from_queue()

        ClientGenerateBatchProcess(service, gen_req, response_handler_wrapper).launch()
        # Don't wait for it to finish! response_handler should take care of
        # notifying when the response is ready. Waiting here increases latency

    def shutdown(self):
        self.service_environment.shutdown()


class LlmMultiProcessServiceManager(LlmServiceManager):
    """
    Manages each service instance in its own process.
    """

    class Instance:
        def __init__(self, instance_num: int, args, max_queue_size: int):
            self.instance_num = instance_num
            self.max_queue_size = max_queue_size
            self.num_outstanding_requests = 0
            self.is_ready_to_board = True
            self.request_queue = JoinableQueue()
            self.response_queue = JoinableQueue()
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.response_map: dict[int, callable] = {}
            self.service_process = LlmServiceProcess(
                args,
                request_queue=self.request_queue,
                response_queue=self.response_queue,
            )

        def start(self):
            self.service_process.start()

            def receive_responses():
                while True:
                    response_packet: Tuple[int, bytes] = self.response_queue.get()
                    self.response_queue.task_done()
                    if response_packet[1] is None:
                        break
                    response_counter = response_packet[0]
                    if response_counter in self.response_map:
                        # print(f"Instance: Received response for request counter: {response_counter}")
                        response_handler = self.response_map.pop(response_counter)
                        response_handler(response_packet[1])
                        self.remove_from_queue()
                    else:
                        logger.warning(
                            f"Instance: Received response for unknown request counter: {response_counter}"
                        )

            self.executor.submit(receive_responses)

        def add_to_queue(self) -> bool:
            if not self.is_ready_to_board:
                return False
            self.num_outstanding_requests += 1
            if self.num_outstanding_requests >= self.max_queue_size:
                print(f"Instance {self.instance_num}: Flight is full")
                self.is_ready_to_board = False
            return True

        def remove_from_queue(self):
            print(
                f"Instance {self.instance_num}: Removing request {self.num_outstanding_requests} of {self.max_queue_size}"
            )
            self.num_outstanding_requests -= 1
            if self.num_outstanding_requests < 0:
                logger.warning(
                    f"Instance {self.instance_num}: Outstanding requests count is negative: {self.num_outstanding_requests}"
                )
            if self.num_outstanding_requests == 0:
                self.is_ready_to_board = True
                print(f"Instance {self.instance_num}: Flight is ready to board again")

        async def send_request(
            self,
            request_counter: int,
            gen_req: GenerateReqInput,
            response_handler: callable,
        ):
            # print(f"Generating multi-process with service: {self.instance_num}")
            # print(f"Enqueuing request {request_counter}"
            #       f" to instance {self.instance_num}")
            print(
                f"Instance {self.instance_num}: Adding request ID {request_counter}, {self.num_outstanding_requests} of {self.max_queue_size}"
            )
            self.request_queue.put((request_counter, gen_req))
            # put shouldn't block, as the caller checked that the queue wasn't full before calling this method
            self.response_map[request_counter] = response_handler

        def shutdown(self):
            print(
                f"Shutting down LlmMultiProcessServiceManager instance {self.instance_num}"
            )
            # Signal the service process to shut down
            self.request_queue.put((0, None))
            # Wait for the service process to finish
            self.service_process.join()
            self.response_queue.join()
            self.executor.shutdown(wait=True)
            print(
                f"LlmMultiProcessServiceManager instance {self.instance_num} shutdown complete"
            )

    def __init__(self, args):
        super().__init__(args)
        self.server_params = LlmServiceEnvironment.get_server_params(args)
        self.model_params = LlmServiceEnvironment.get_model_params(args.model_config)
        self.num_instances = self.server_params.instances
        max_queue_size = max(self.model_params.decode_batch_sizes)

        self.instances: List[LlmMultiProcessServiceManager.Instance] = [
            LlmMultiProcessServiceManager.Instance(i, args, max_queue_size)
            for i in range(self.num_instances)
        ]

    def start(self):
        for instance in self.instances:
            instance.start()

    async def send_request(self, gen_req: GenerateReqInput, response_handler: callable):
        self.request_counter += 1

        # Save global request counter before the await, or else it may change
        # if another request is sent
        request_counter = self.request_counter

        # instance_num = self.get_next_instance_num()
        # Fill up the queue of the current instance first, so that the batcher
        # can dispatch a batch ASAP
        instance_num = self.cur_instance_num
        first_tried_instance_num = instance_num
        while not self.instances[instance_num].add_to_queue():
            instance_num = self.get_next_instance_num()
            if instance_num == first_tried_instance_num:
                await asyncio.sleep(0.001)
        self.cur_instance_num = instance_num

        await self.instances[instance_num].send_request(
            request_counter, gen_req, response_handler
        )

    def shutdown(self):
        for instance in self.instances:
            instance.shutdown()
