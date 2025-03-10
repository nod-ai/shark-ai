# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import threading

import uvicorn.logging

# Import first as it does dep checking and reporting.
from shortfin import ProgramIsolation
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
    # parameters are loaded with `iree_io_parameters_module_create`
    parser.add_argument(
        "--parameters",
        type=Path,
        nargs="*",
        help="Parameter archives to load (supports: gguf, irpa, safetensors).",
        metavar="FILE",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["local-task", "hip", "amdgpu"],
        help="Device to serve on; e.g. local-task, hip. Same options as `iree-run-module --device` ",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        nargs="*",
        default=None,
        help="Device IDs visible to the system builder. Defaults to None (full visibility). Can be an index or a sf device id like amdgpu:0:0@0",
    )
    parser.add_argument(
        "--isolation",
        type=str,
        default="per_call",
        choices=[isolation.name.lower() for isolation in ProgramIsolation],
        help="Concurrency control -- How to isolate programs.",
    )
    parser.add_argument(
        "--amdgpu_async_allocations",
        action="store_true",
        help="Enable asynchronous allocations for amdgpu device contexts.",
    )
    parser.add_argument(
        "--amdgpu_allocators",
        default=None,
        help="Allocator to use during VMFB invocation.",
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
        "--n_beams",
        type=int,
        default=1,
        help="""The number of beams to use during the decode sequence.\n\n

        If `n_beams == 1`, `greedy` decode strategy is used.\n
        If `n_beams > 1`, `beam_search` decode strategy is used.\n
        """,
    )
    return parser.parse_args(argv)


def main(argv, log_config=uvicorn.config.LOGGING_CONFIG):
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
        port=args.port,
        log_config=log_config,
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    from shortfin.support.logging_setup import configure_main_logger
    import yappi

    logger = configure_main_logger("server")

    # Configure yappi for multi-threaded and async code profiling
    yappi.set_clock_type(
        "cpu"
    )  # Use CPU time (alternatives: "wall" for wall-clock time)

    # Safer context ID callback that checks if event loop exists first
    def context_id_callback():
        try:
            # Check if there's a running event loop before getting current task
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.current_task(loop)
                if task:
                    return id(task)
        except RuntimeError:
            # No event loop, that's ok
            pass
        # Default to thread ID
        return id(threading.current_thread())

    yappi.set_context_id_callback(context_id_callback)

    # Start profiling
    yappi.start(builtins=True)

    try:
        # Run your function
        main(
            sys.argv[1:],
            # Make logging defer to the default shortfin logging config.
            log_config=UVICORN_LOG_CONFIG,
        )
    finally:
        # Stop profiling and save to file
        yappi.stop()

        # Write profiling results to file
        stats = yappi.get_func_stats()
        stats.save("shortfin_llm_server.prof", type="pstat")  # Same format as cProfile

        # Optionally save in yappi format for more detailed analysis
        stats.save("shortfin_llm_server.yappi", type="callgrind")

        # Log where the profiles were saved
        logger.info("Profiling data saved to:")
        logger.info("  - shortfin_llm_server.prof (pstat format)")
        logger.info("  - shortfin_llm_server.yappi (callgrind format)")
