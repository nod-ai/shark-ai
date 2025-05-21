# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import asyncio
import json
import logging
import sys
import time
import numpy as np


# Import first as it does dep checking and reporting.
from pathlib import Path
from typing import List, Optional
from shortfin.support.logging_setup import configure_main_logger
from shortfin.support.responder import AbstractResponder

from .components.generate import ClientGenerateBatchProcess
from .components.io_struct import GenerateReqInput, SamplingParams
from .components.lifecycle import ShortfinLlmLifecycleManager
from .server import add_service_args


logger = logging.getLogger(__name__)


def add_input_args(parser):
    group = parser.add_argument_group("Input Source", "Inputs to select from")
    group = group.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file")


def add_cli_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Perform a benchmarking run for throughput",
    )
    parser.add_argument(
        "--benchmark_tasks",
        type=int,
        default=None,
        help="Workload size to benchmark with",
    )
    parser.add_argument(
        "--decode_steps",
        type=int,
        default=5,
        help="The number of decode steps to execute",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        help="Temperature value to use for `offline` generation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        help="Top K value to use for `offline` generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        required=False,
        help="Top P value to use for `offline` generation.",
    )
    parser.add_argument(
        "--workers_offline",
        type=int,
        default=1,
        help="Number of workers to use when running in `offline` mode.",
    )


def parse_args(argv):
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    add_input_args(parser)
    add_cli_args(parser)
    return parser.parse_args(argv)


def process_inputs(args):

    if args.prompt:
        if args.benchmark and args.benchmark_tasks is not None:
            prompts = [args.prompt] * args.benchmark_tasks
        return prompts

    return json.load(open(args.prompt_file, "r"))


class Timer:
    def __init__(self, name: str):
        self._name = name
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self):
        self._start = time.perf_counter()
        logger.info(f"{self._name} start time: {self._start}")

    def end(self):
        self._end = time.perf_counter()
        logger.info(f"{self._name} end time: {self._end}")

    def elapsed(self):
        if self._end is None and self._start is not None:
            return time.perf_counter() - self._start
        if self._end is not None and self._start is not None:
            return self._end - self._start
        return 0


class CliResponder(AbstractResponder):
    _idx: int = 0

    def __init__(self):
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self.response = asyncio.Future(loop=self._loop)
        self.responded = False
        self.idx = self._get_idx()
        self.name = f"CliResponder-{self.idx}"
        self._timer = Timer(self.name)

    @classmethod
    def _get_idx(cls):
        cls._idx += 1
        return cls._idx

    def elapsed(self):
        return self._timer.elapsed()

    def start_response(self):
        self._timer.start()

    def ensure_response(self):
        self._timer.end()

    def send_response(self, response):
        logger.info(f"{self.name} Sending response")
        assert not self.responded, "Response already sent"
        if self._loop.is_closed():
            raise IOError("Web server is shut down")
        self.responded = True
        self._loop.call_soon_threadsafe(self.response.set_result, response)

    def stream_start(self, **kwargs):
        raise Exception("Streaming not supported")

    def stream_part(self, content):
        raise Exception("Streaming not supported")


async def main(argv):
    args = parse_args(argv)
    if args.tokenizer_config_json is None:
        # this is only used for the EOS token
        logger.info("Argument `--tokenizer_config_json` is not provided")
        logger.info("Inferring tokenizer config path from tokenizer path")
        inferred_tokenizer_config_path = args.tokenizer_json.with_name(
            args.tokenizer_json.stem + "_config.json"
        )
        args.tokenizer_config_json = inferred_tokenizer_config_path

    logger.info("Setting up service")
    lifecycle_manager = ShortfinLlmLifecycleManager(args)
    service = lifecycle_manager.services["default"]
    service.start()

    sampling_params = SamplingParams(max_completion_tokens=args.decode_steps)
    if getattr(args, "temperature", None) is not None:
        sampling_params.temperature = args.temperature
    if getattr(args, "top_k", None) is not None:
        sampling_params.top_k = args.top_k
    if getattr(args, "top_p", None) is not None:
        sampling_params.top_p = args.top_p

    prompts = process_inputs(args)

    class Task:
        def __init__(self, prompt):
            self.prompt = prompt
            self.responder: Optional[CliResponder] = None
            self.result: Optional[str] = None

        def runtime(self):
            if self.responder is not None:
                return self.responder.elapsed()
            return 0

    logger.info(f"Setting up a tasklist of {len(prompts)} items")
    tasks: List[Task] = []
    for p in prompts:
        task = Task(p)
        tasks.append(task)

    async def worker(name, queue, fiber):
        while True:
            task: Task = await queue.get()
            responder = CliResponder()
            gen_req = GenerateReqInput(
                text=task.prompt, sampling_params=sampling_params
            )
            ClientGenerateBatchProcess(
                service, gen_req, responder, fiber=fiber
            ).launch()
            await responder.response
            task.responder = responder
            task.result = responder.response.result()
            queue.task_done()

    logger.info(f"Setting up {args.workers_offline} workers")
    workers = []
    queue = asyncio.Queue()
    for i in range(args.workers_offline):
        name = f"worker-{i}"
        workerr = service.sysman.ls.create_worker(name)
        fiber = service.sysman.ls.create_fiber(workerr)
        w = asyncio.create_task(worker(name, queue, fiber))
        workers.append(w)

    logger.info(f"Processing tasks")

    global_timer = Timer("global")
    global_timer.start()
    for t in tasks:
        queue.put_nowait(t)

    await queue.join()
    global_timer.end()

    for w in workers:
        w.cancel()

    if args.benchmark:
        total_time = global_timer.elapsed()
        reqs = len(prompts) / total_time

        print(f"Requests per second: {reqs:2f}")
        latencies = [s.runtime() for s in tasks]
        print(
            f"Latencies: av: {np.mean(latencies)}, min: {np.min(latencies)}, max: {np.max(latencies)}, median: {np.median(latencies)}, sd: {np.std(latencies)}"
        )

    logger.info(f"Shutting down service")
    service.shutdown()


if __name__ == "__main__":
    configure_main_logger("cli")
    asyncio.run(main(sys.argv[1:]))
