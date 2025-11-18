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

# Import first as it does dep checking and reporting.
from shortfin.interop.fastapi import FastAPIResponder
from shortfin.support.logging_setup import native_handler

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .components.generate import ClientGenerateBatchProcess
from .components.config_struct import ModelParams
from .components.io_struct import GenerateReqInput
from .components.manager import SDXLSystemManager
from .components.service import SDXLGenerateService
from .components.tokenizer import Tokenizer


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
    model_config, topology_config, flagfile, tuning_spec, args = get_configs(args)
    sysman = SDXLSystemManager(
        args.device, args.device_ids, args.amdgpu_async_allocations
    )
    return sysman, model_config, flagfile, tuning_spec


def configure_service(args, sysman, model_config, flagfile, tuning_spec):
    # Setup each service we are hosting.
    tokenizers = []
    for idx, tok_name in enumerate(args.tokenizers):
        subfolder = f"tokenizer_{idx + 1}" if idx > 0 else "tokenizer"
        tokenizers.append(Tokenizer.from_pretrained(tok_name, subfolder))
    model_params = ModelParams.load_json(model_config)
    vmfbs, params = get_modules(args, model_config, flagfile, tuning_spec)

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


def get_configs(args):
    # Returns one set of config artifacts.
    modelname = "sdxl"
    model_config = args.model_config if args.model_config else None
    topology_config = None
    tuning_spec = None
    flagfile = args.flagfile if args.flagfile else None
    topology_inp = args.topology if args.topology else "spx_single"
    cfg_builder_args = [
        sys.executable,
        "-m",
        "iree.build",
        os.path.join(THIS_DIR, "components", "config_artifacts.py"),
        f"--target={args.target}",
        f"--output-dir={args.artifacts_dir}",
        f"--model={modelname}",
        f"--topology={topology_inp}",
    ]
    outs = subprocess.check_output(cfg_builder_args).decode()
    outs_paths = outs.splitlines()
    for i in outs_paths:
        if model_config:
            if str(model_config) in i and not os.path.exists(model_config):
                model_config = i
        elif not model_config and "sdxl_config" in i:
            model_config = i
        elif "topology" in i and args.topology:
            topology_config = i
        elif "flagfile" in i and not args.flagfile:
            flagfile = i
        elif "attention_and_matmul_spec" in i and args.use_tuned:
            tuning_spec = i

    if not os.path.exists(model_config):
        raise ValueError(f"Model config not found at {model_config}.")

    if args.use_tuned and args.tuning_spec:
        tuning_spec = os.path.abspath(args.tuning_spec)

    if topology_config:
        with open(topology_config, "r") as f:
            contents = [line.rstrip() for line in f]
        for spec in contents:
            if "--" in spec:
                arglist = spec.strip("--").split("=")
                arg = arglist[0]
                if len(arglist) > 2:
                    value = arglist[1:]
                    for val in value:
                        try:
                            val = int(val)
                        except ValueError:
                            val = val
                elif len(arglist) == 2:
                    value = arglist[-1]
                    try:
                        value = int(value)
                    except ValueError:
                        value = value
                else:
                    # It's a boolean arg.
                    value = True
                setattr(args, arg, value)
            else:
                # It's an env var.
                arglist = spec.split("=")
                os.environ[arglist[0]] = arglist[1]
    return model_config, topology_config, flagfile, tuning_spec, args


def get_modules(args, model_config, flagfile, td_spec):
    # TODO: Move this out of server entrypoint
    mod_params = ModelParams.load_json(model_config)

    vmfbs = {}
    params = {}
    model_flags = {}
    for submodel in mod_params.module_names.keys():
        vmfbs[submodel] = {}
        model_flags[submodel] = []
        for bs in mod_params.batch_sizes[submodel]:
            vmfbs[submodel][bs] = []
        if submodel != "scheduler":
            params[submodel] = []
    model_flags["all"] = args.compile_flags

    if flagfile:
        with open(flagfile, "r") as f:
            contents = [line.rstrip() for line in f]
        flagged_model = "all"
        for elem in contents:
            match = [keyw in elem for keyw in model_flags.keys()]
            if any(match) or "--" not in elem:
                flagged_model = elem
            elif flagged_model in model_flags:
                model_flags[flagged_model].extend([elem])
    if td_spec:
        for key in model_flags.keys():
            if key in ["unet", "punet", "scheduled_unet"]:
                model_flags[key].extend(
                    [f"--iree-codegen-transform-dialect-library={td_spec}"]
                )
    filenames = []
    builder_env = os.environ.copy()
    builder_env["IREE_BUILD_MP_CONTEXT"] = "fork"
    for modelname in vmfbs.keys():
        ireec_args = model_flags["all"] + model_flags[modelname]
        ireec_extra_args = " ".join(ireec_args)
        builder_args = [
            sys.executable,
            "-m",
            "iree.build",
            os.path.join(THIS_DIR, "components", "builders.py"),
            f"--model-json={model_config}",
            f"--target={args.target}",
            f"--splat={args.splat}",
            f"--build-preference={args.build_preference}",
            f"--output-dir={args.artifacts_dir}",
            f"--model={modelname}",
            f"--force-update={args.force_update}",
            f"--iree-hal-target-device={args.device}",
            f"--iree-hip-target={args.target}",
            f"--iree-compile-extra-args={ireec_extra_args}",
        ]
        logger.info(f"Preparing runtime artifacts for {modelname}...")
        logger.info(f"Builder args: {' '.join(builder_args)}")
        logger.debug(
            "COMMAND LINE EQUIVALENT: " + " ".join([str(argn) for argn in builder_args])
        )
        output = subprocess.check_output(builder_args, env=builder_env).decode()

        output_paths = output.splitlines()
        for path in output_paths:
            if not any(x in path for x in [".mlir", ".vmfb", ".irpa"]):
                output_paths.remove(path)
            if "irpa" in path:
                params[modelname].append(path)
                output_paths.remove(path)
        filenames.extend(output_paths)
    for name in filenames:
        for key in vmfbs.keys():
            for bs in vmfbs[key].keys():
                if key in name.lower() and f"_bs{bs}_" in name.lower():
                    if "vmfb" in name:
                        vmfbs[key][bs].extend([name])
    return vmfbs, params


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
        "--device",
        type=str,
        required=True,
        choices=["local-task", "hip", "amdgpu"],
        help="Primary inferencing device.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        default="gfx942",
        choices=["gfx90a", "gfx942", "gfx1100", "gfx1201", "gfx950"],
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
        artdir = home / ".cache" / "amdshark"
        args.artifacts_dir = str(artdir)
    else:
        args.artifacts_dir = os.path.abspath(args.artifacts_dir)

    global sysman
    sysman, model_config, flagfile, tuning_spec = configure_sys(args)
    configure_service(args, sysman, model_config, flagfile, tuning_spec)
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
