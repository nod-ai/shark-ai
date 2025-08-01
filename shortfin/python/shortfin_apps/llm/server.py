# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
from pathlib import Path
import sys

import uvicorn.logging

# Import first as it does dep checking and reporting.
from shortfin import ProgramIsolation

import uvicorn

from .application import get_app
from .components.lifecycle import ShortfinLlmLifecycleManager
from .components.token_selection_strategy import TokenSelectionStrategy
from ..utils import get_system_args

logger = logging.getLogger(__name__)

UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": "[{asctime}] {message}",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "{",
            "use_colors": True,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


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
        "--program_isolation",
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
        "--use_beam_search",
        action="store_true",
        default=False,
        help="Use beam search for decoding.",
    )
    parser.add_argument(
        "--use_new_decoder",
        action="store_true",
        default=False,
        help="Use the new decoder infrastructure.",
    )
    parser.add_argument(
        "--use_native_impls",
        action="store_true",
        default=False,
        help="Use native implementations for token selection.",
    )


def parse_args(argv):
    parser = argparse.ArgumentParser()
    add_service_args(parser)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="Root path to use for installing behind path based proxy.",
    )
    parser.add_argument(
        "--timeout-keep-alive", type=int, default=5, help="Keep alive timeout"
    )
    return parser.parse_args(argv)


def run_server(argv, log_config=uvicorn.config.LOGGING_CONFIG, port: int | None = None):
    args = parse_args(argv)
    if args.tokenizer_config_json is None:
        # this is only used for the EOS token
        logging.info("Argument `--tokenizer_config_json` is not provided")
        logging.info("Inferring tokenizer config path from tokenizer path")
        inferred_tokenizer_config_path = args.tokenizer_json.with_name(
            args.tokenizer_json.stem + "_config.json"
        )
        args.tokenizer_config_json = inferred_tokenizer_config_path

    lifecycle_manager = ShortfinLlmLifecycleManager(args)

    uvicorn.run(
        get_app(lifecycle_manager.fastapi_lifespan),
        host=args.host,
        port=port or args.port,
        log_config=log_config,
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    from shortfin.support.logging_setup import configure_main_logger

    logger = configure_main_logger("server")
    run_server(
        sys.argv[1:],
        # Make logging defer to the default shortfin logging config.
        log_config=UVICORN_LOG_CONFIG,
    )
