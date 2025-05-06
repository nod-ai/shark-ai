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
from .cli import add_service_args

import uvicorn

from .application import get_app
from .components.lifecycle import ShortfinLlmLifecycleManager


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


def _check_for_per_fiber_bug(args):
    """This is a temporary check to enable multi-worker and multi-fiber-per-worker
    for performance benefits.

    TODO: https://github.com/nod-ai/shark-ai/issues/1284

    Raises:
        NotImplementedError: Raises if per fiber isolation is used with multiple fibers per worker.
    """
    isolation = args.program_isolation
    fibers_per_worker = args.fibers_per_worker

    if isolation == ProgramIsolation.PER_FIBER.name.lower() and fibers_per_worker > 1:
        raise NotImplementedError(
            "Per fiber isolation does not currently support multiple fibers per worker. "
            "Please set `--fibers_per_worker` to 1.\n"
            "See: https://github.com/nod-ai/shark-ai/issues/1284"
        )


def run_server(argv, log_config=uvicorn.config.LOGGING_CONFIG, port: int | None = None):
    args = parse_args(argv)
    _check_for_per_fiber_bug(args)
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
