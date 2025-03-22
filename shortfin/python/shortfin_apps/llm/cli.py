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


# Import first as it does dep checking and reporting.
from pathlib import Path
from shortfin import ProgramIsolation
from shortfin.support.responder import AbstractResponder

from .components.generate import ClientGenerateBatchProcess
from .components.io_struct import GenerateReqInput
from .components.lifecycle import ShortfinLlmLifecycleManager
from ..utils import get_system_args


logger = logging.getLogger(__name__)


def add_input_args(parser):
    group = parser.add_argument_group("Input Source", "Inputs to select from")
    group = group.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file")


def add_service_args(parser: argparse.ArgumentParser):
    get_system_args(parser)

    parser.add_argument(
        "--tokenizer_json",
        type=Path,
        required=True,
        help="Path to a tokenizer.json file",
    )
    parser.add_argument(
        "--tokenizer_config_json",
        type=Path,
        required=False,
        help="Path to a tokenizer_config json file",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        required=True,
        help="Path to the model config file",
    )
    parser.add_argument(
        "--vmfb",
        type=Path,
        required=True,
        help="Model VMFB to load",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        nargs="*",
        help="Parameter archives to load (supports: gguf, irpa, safetensors).",
        metavar="FILE",
    )
    parser.add_argument(
        "--isolation",
        type=str,
        default="per_call",
        choices=[isolation.name.lower() for isolation in ProgramIsolation],
        help="Concurrency control -- How to isolate programs.",
    )
    parser.add_argument(
        "--server_config",
        type=Path,
        help="Path to server configuration file",
    )
    parser.add_argument(
        "--prefix_sharing_algorithm",
        type=str,
        choices=["none", "trie"],
        help="Algorithm to use for prefix sharing in KV cache",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use during decode sequence. Defaults to `1`.",
    )
    parser.add_argument(
        "--token_selection_strategy",
        type=str,
        choices=["greedy", "multi_greedy"],
        default="greedy",
        help="Strategy to use when selecting tokens during generation. Defaults to `greedy`.",
    )
    parser.add_argument(
        "--decode_steps",
        type=int,
        default=5,
        help="The number of decode steps to execute",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent requests that should be running",
    )
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


def parse_args(argv):
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    add_input_args(parser)

    return parser.parse_args(argv)


def process_inputs(args):
    if args.prompt:
        prompts = [args.prompt]
        if args.benchmark and args.benchmark_tasks is not None:
            prompts = prompts * args.benchmark_tasks
        return prompts

    return json.load(open(args.prompt_file, "r"))


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def start(self):
        self._start = time.perf_counter()

    def end(self):
        self._end = time.perf_counter()

    def elapsed(self):
        return self._end - self._start


class CliResponder(AbstractResponder):
    def __init__(self):
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self.response = asyncio.Future(loop=self._loop)
        self.responded = False
        self.timer = Timer()

    def start_response(self):
        self.timer.start()

    def ensure_response(self):
        self.timer.end()

    def send_response(self, response):
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
        logging.info("Argument `--tokenizer_config_json` is not provided")
        logging.info("Inferring tokenizer config path from tokenizer path")
        inferred_tokenizer_config_path = args.tokenizer_json.with_name(
            args.tokenizer_json.stem + "_config.json"
        )
        args.tokenizer_config_json = inferred_tokenizer_config_path

    logger.info(msg="Setting up service", level=logging.INFO)
    lifecycle_manager = ShortfinLlmLifecycleManager(args)
    service = lifecycle_manager.services["default"]
    service.start()

    sampling_params = {"max_completion_tokens": args.decode_steps}

    prompts = process_inputs(args)

    class Task:
        def __init__(self, prompt):
            self.prompt = prompt
            self.responder = None

        def runtime(self):
            return self.responder.timer.elapsed()

    logger.log(msg=f"Setting up a tasklist of {len(prompts)} items", level=logging.INFO)
    queue = asyncio.Queue()
    tasks = []
    for p in prompts:
        task = Task(p)
        tasks.append(task)
        queue.put_nowait(task)

    async def worker(name, queue):
        while True:
            task = await queue.get()
            responder = CliResponder()
            gen_req = GenerateReqInput(
                text=task.prompt, sampling_params=sampling_params
            )
            ClientGenerateBatchProcess(service, gen_req, responder).launch()
            await responder.response
            task.responder = responder
            task.result = responder.response.result()
            queue.task_done()

    global_timer = Timer()
    global_timer.start()

    logger.log(msg=f"Setting up {args.workers} workers", level=logging.INFO)
    workers = []
    for i in range(args.workers):
        w = asyncio.create_task(worker(f"worker-{i}", queue))
        workers.append(w)

    logger.log(msg=f"Processing tasks", level=logging.INFO)
    await queue.join()
    global_timer.end()

    for w in workers:
        w.cancel()

    if args.benchmark:
        latency_sum = sum([s.runtime() for s in tasks])
        latency_avg = latency_sum / len(tasks)
        total_time = global_timer.elapsed()
        reqs = len(prompts) / total_time

        print(f"Requests per second: {reqs:2f}")
        print(f"AverageLatency:      {latency_avg:2f}")

    logger.log(msg=f"Shutting down service", level=logging.INFO)
    service.shutdown()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
