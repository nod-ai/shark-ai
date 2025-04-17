# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
import argparse
import logging
from pathlib import Path
import sys
import os
import copy
import subprocess
from contextlib import asynccontextmanager
import uvicorn
import importlib.util

# Import first as it does dep checking and reporting.
from shortfin.interop.fastapi import FastAPIResponder
from shortfin.support.logging_setup import native_handler

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from shortfin_apps.utilities.artifacts import fetch_modules

from .components.generate import ClientGenerateBatchProcess
from .components.config_struct import ModelParams
from .components.io_struct import GenerateReqInput
from .components.manager import SDXLSystemManager
from .components.service import SDXLGenerateService
from .components.tokenizer import Tokenizer
from .components.config_artifacts import get_configs

if spec := importlib.util.find_spec("sharktank") is None:
    sharktank_installed = False
else:
    sharktank_installed = True


logger = logging.getLogger("shortfin-sd")
logger.addHandler(native_handler)
logger.propagate = False

THIS_DIR = Path(__file__).parent

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    sysman.start()
    try:
        for service_name, service in services.items():
            logger.info("Initializing service '%s':", service_name)
            logger.info(str(service))
            service.start()
    except:
        sysman.shutdown()
        raise
    yield
    try:
        for service_name, service in services.items():
            logger.info("Shutting down service '%s'", service_name)
            service.shutdown()
    finally:
        sysman.shutdown()


sysman: SDXLSystemManager
services: dict[str, Any] = {}
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


async def generate_request(gen_req: GenerateReqInput, request: Request):
    service = services["sd"]
    gen_req.post_init()
    responder = FastAPIResponder(request)
    ClientGenerateBatchProcess(service, gen_req, responder).launch()
    return await responder.response


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


# -------- MIDDLEWARE --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def configure_sys(args) -> SDXLSystemManager:
    # Setup system (configure devices, etc).
    artifacts = get_configs(
        args.service,
        args.target,
        args.artifacts_dir,
        args.model_config,
        args.tuning_spec,
        args.flagfile,
    )
    sysman = SDXLSystemManager(
        args.device, args.device_ids, args.amdgpu_async_allocations
    )
    return sysman, artifacts


def configure_service(args, vmfbs, params, sysman, model_config, flagfile, tuning_spec):
    # Setup each service we are hosting.
    tokenizers = []
    for idx, tok_name in enumerate(args.tokenizers):
        subfolder = f"tokenizer_{idx + 1}" if idx > 0 else "tokenizer"
        tokenizers.append(Tokenizer.from_pretrained(tok_name, subfolder))
    model_params = ModelParams.load_json(model_config)

    sm = SDXLGenerateService(
        name="sd",
        sysman=sysman,
        tokenizers=tokenizers,
        model_params=model_params,
        fibers_per_device=args.fibers_per_device,
        workers_per_device=args.workers_per_device,
        prog_isolation=args.isolation,
        show_progress=args.show_progress,
        trace_execution=args.trace_execution,
        splat=args.splat,
    )
    for key, vmfb_dict in vmfbs.items():
        for bs in vmfb_dict.keys():
            for vmfb in vmfb_dict[bs]:
                sm.load_inference_module(vmfb, component=key, batch_size=bs)
    for key, datasets in params.items():
        sm.load_inference_parameters(*datasets, parameter_scope="model", component=key)
    services[sm.name] = sm
    return sysman


def is_port_valid(port):
    max_port = 65535
    if port < 1 or port > max_port:
        print(
            f"Error: Invalid port specified ({port}), expected a value between 1 and {max_port}"
        )
        return False
    return True


def main(argv, log_config=UVICORN_LOG_CONFIG):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--timeout-keep-alive", type=int, default=5, help="Keep alive timeout"
    )
    parser.add_argument(
        "--service",
        type=str,
        default="sdxl",
        choices=["sdxl"],
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["local-task", "amdgpu"],
        help="Primary inferencing device.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        default="gfx942",
        choices=["gfx90a", "gfx942", "gfx1100", "gfx1201"],
        help="Primary inferencing device LLVM target arch.",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        nargs="*",
        default=None,
        help="Device IDs visible to the system builder. Defaults to None (full visibility). Can be an IREE index or a device id like GPU-66613339-3934-3261-3131-396338323735.",
    )
    parser.add_argument(
        "--tokenizers",
        type=Path,
        nargs="*",
        default=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
        help="HF repo from which to load tokenizer(s).",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        help="Path to the model config file. If None, defaults to i8 punet, batch size 1",
    )
    parser.add_argument(
        "--workers_per_device",
        type=int,
        default=1,
        help="Concurrency control -- how many fibers are created per device to run inference.",
    )
    parser.add_argument(
        "--fibers_per_device",
        type=int,
        default=1,
        help="Concurrency control -- how many fibers are created per device to run inference.",
    )
    parser.add_argument(
        "--isolation",
        type=str,
        default="per_call",
        choices=["per_fiber", "per_call", "none"],
        help="Concurrency control -- How to isolate programs.",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="enable tqdm progress for unet iterations.",
    )
    parser.add_argument(
        "--trace_execution",
        action="store_true",
        help="Enable tracing of program modules.",
    )
    parser.add_argument(
        "--amdgpu_async_allocations",
        action="store_true",
        help="Enable asynchronous allocations for amdgpu device contexts.",
    )
    parser.add_argument(
        "--splat",
        action="store_true",
        help="Use splat (empty) parameter files, usually for testing.",
    )
    parser.add_argument(
        "--build_preference",
        type=str,
        choices=["compile", "precompiled", "export"],
        default="precompiled",
        help="Specify preference for builder artifact generation.",
    )
    parser.add_argument(
        "--compile_flags",
        type=str,
        nargs="*",
        default=[],
        help="extra compile flags for all compile actions. For fine-grained control, use flagfiles.",
    )
    parser.add_argument(
        "--flagfile",
        type=Path,
        help="Path to a flagfile to use for SDXL. If not specified, will use latest flagfile from azure.",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=None,
        help="Path to local artifacts cache.",
    )
    parser.add_argument(
        "--tuning_spec",
        type=str,
        default=None,
        help="Path to transform dialect spec if compiling an executable with tunings.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        choices=["spx_single", "cpx_single", "spx_multi", "cpx_multi"],
        help="Use one of four known performant preconfigured device/fiber topologies.",
    )
    parser.add_argument(
        "--use_tuned",
        type=int,
        default=1,
        help="Use tunings for attention and matmul ops. 0 to disable.",
    )
    parser.add_argument(
        "--force_update",
        default=False,
        help="Force update model artifacts starting from the specified build preference.",
    )
    args = parser.parse_args(argv)
    if not is_port_valid(args.port):
        exit(3)

    if not args.artifacts_dir:
        home = Path.home()
        artdir = home / ".cache" / "shark"
        args.artifacts_dir = str(artdir)
    else:
        args.artifacts_dir = os.path.abspath(args.artifacts_dir)

    global sysman
    sysman, artifacts = configure_sys(args)

    vmfbs, params, needed = fetch_modules(
        args.target,
        args.device,
        "sdxl",
        artifacts["model_config"],
        args.artifacts_dir,
        args.splat,
    )
    if (needed or args.force_update) and sharktank_installed:
        from sharktank.pipelines.sdxl.builder import get_modules

        vmfbs, params = get_modules(
            vmfbs,
            params,
            args.target,
            args.device,
            artifacts["model_config"],
            artifacts["flagfile"],
            artifacts["tuning_spec"],
            args.compile_flags,
            args.artifacts_dir,
            args.splat,
            args.build_preference,
            args.force_update,
        )

    elif (needed or args.force_update) and not sharktank_installed:
        raise FileNotFoundError(str(needed.values()))

    configure_service(
        args,
        vmfbs,
        params,
        sysman,
        artifacts["model_config"],
        artifacts["flagfile"],
        artifacts["tuning_spec"],
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=log_config,
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)
    main(
        sys.argv[1:],
        # Make logging defer to the default shortfin logging config.
        log_config=UVICORN_LOG_CONFIG,
    )
