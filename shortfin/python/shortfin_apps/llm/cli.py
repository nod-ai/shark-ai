# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

# Import first as it does dep checking and reporting.
from shortfin import ProgramIsolation
from shortfin.support.responder import AbstractResponder

from .components.generate import ClientGenerateBatchProcess
from .components.io_struct import GenerateReqInput
from .components.lifecycle import ShortfinLlmLifecycleManager
from ..utils import get_system_args


logger = logging.getLogger(__name__)


def add_input_args(parser):
    group = parser.add_argument_group("Input Source", "Inputs to select from")
    group = group.add_mutually_exclusive_group()
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")


def add_service_args(parser):
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


def parse_args(argv):
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    add_input_args(parser)

    return parser.parse_args(argv)


def process_inputs(args):
    if args.prompt:
        return [args.prompt]
    return json.load(open(args.prompt_file, "r"))


class CliResponder(AbstractResponder):
    def __init__(self):
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self.response = asyncio.Future(loop=self._loop)
        self.responded = False

    def ensure_response(self):
        pass

    def send_response(self, response):
        assert not self.responded, "Response already sent"
        if self._loop.is_closed():
            raise IOError("Web server is shut down")
        self.responded = True
        self._loop.call_soon_threadsafe(self.response.set_result, response)

    def start_response(self, **kwargs):
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

    sampling_params = {"max_completion_tokens": 5}

    prompts = process_inputs(args)

    responders = []
    for prompt in prompts:
        logger.log(msg=f'Submitting request for prompt "{prompt}"', level=logging.INFO)
        gen_req = GenerateReqInput(text=prompt, sampling_params=sampling_params)
        responder = CliResponder()

        async def submit():
            ClientGenerateBatchProcess(service, gen_req, responder).launch()
            return responder

        await submit()
        responders.append(responder)

    await asyncio.gather(*[r.response for r in responders])

    for responder in responders:
        print(responder.response.result().decode())

    logger.log(msg=f"Shutting down service", level=logging.INFO)
    service.shutdown()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
