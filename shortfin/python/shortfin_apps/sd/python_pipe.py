# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
import argparse
import logging
import asyncio
from pathlib import Path
import sys
import time
import os
import copy
import subprocess
from contextlib import asynccontextmanager
import uvicorn

# Import first as it does dep checking and reporting.
from shortfin.interop.fastapi import FastAPIResponder
from shortfin.support.logging_setup import native_handler
import shortfin as sf

from fastapi import FastAPI, Request, Response

from .components.generate import GenerateImageProcess
from .components.messages import InferenceExecRequest, InferencePhase
from .components.config_struct import ModelParams
from .components.io_struct import GenerateReqInput
from .components.manager import SystemManager
from .components.service import GenerateService, InferenceExecutorProcess
from .components.tokenizer import Tokenizer


logger = logging.getLogger("shortfin-sd")
logger.addHandler(native_handler)
logger.propagate = False

THIS_DIR = Path(__file__).parent

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
        if "sdxl_config" in i and not args.model_config:
            model_config = i
        elif "topology" in i and args.topology:
            topology_config = i
        elif "flagfile" in i and not args.flagfile:
            flagfile = i
        elif "attention_and_matmul_spec" in i and args.use_tuned:
            tuning_spec = i

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
    vmfbs = {"clip": [], "unet": [], "vae": [], "scheduler": []}
    params = {"clip": [], "unet": [], "vae": []}
    model_flags = copy.deepcopy(vmfbs)
    model_flags["all"] = args.compile_flags

    if flagfile:
        with open(flagfile, "r") as f:
            contents = [line.rstrip() for line in f]
        flagged_model = "all"
        for elem in contents:
            match = [keyw in elem for keyw in model_flags.keys()]
            if any(match):
                flagged_model = elem
            else:
                model_flags[flagged_model].extend([elem])
    if td_spec:
        model_flags["unet"].extend(
            [f"--iree-codegen-transform-dialect-library={td_spec}"]
        )

    filenames = []
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
            f"--iree-hal-target-device={args.device}",
            f"--iree-hip-target={args.target}",
            f"--iree-compile-extra-args={ireec_extra_args}",
        ]
        logger.info(f"Preparing runtime artifacts for {modelname}...")
        logger.debug(
            "COMMAND LINE EQUIVALENT: " + " ".join([str(argn) for argn in builder_args])
        )
        output = subprocess.check_output(builder_args).decode()

        output_paths = output.splitlines()
        filenames.extend(output_paths)
    for name in filenames:
        for key in vmfbs.keys():
            if key in name.lower():
                if any(x in name for x in [".irpa", ".safetensors", ".gguf"]):
                    params[key].extend([name])
                elif "vmfb" in name:
                    vmfbs[key].extend([name])
    return vmfbs, params

class MicroSDXLServer(sf.Process):

    def __init__(self, args, service):
        super().__init__(fiber=service.fibers[0])
        self.service = service

        self.args = args
        self.exec = None
        self.imgs = None

    async def run(
        self
    ):
        args = self.args
        self.exec = InferenceExecRequest(
            args.prompt,
            args.neg_prompt,
            1024,
            1024,
            args.steps,
            args.guidance_scale,
            args.seed,
        )
        self.exec.phases[InferencePhase.POSTPROCESS]["required"] = False
        while len(self.service.idle_fibers) == 0:
            time.sleep(0.5)
            print("All fibers busy...")
        fiber = self.service.idle_fibers.pop()
        fiber_idx = self.service.fibers.index(fiber)
        worker_idx = self.service.get_worker_index(fiber)
        exec_process = InferenceExecutorProcess(self.service, fiber)
        if self.service.prog_isolation != sf.ProgramIsolation.PER_FIBER:
                self.service.idle_fibers.add(fiber)
        exec_process.exec_requests.append(self.exec)
        exec_process.launch()
        await asyncio.gather(exec_process)
        imgs = []
        await self.exec.done

        for req in exec_process.exec_requests:
            imgs.append(req.image_array)
        
        self.imgs = imgs
        return

class Main:
    def __init__(self, sysman):
        self.sysman = sysman

    async def main(self, args):
        tokenizers = []
        for idx, tok_name in enumerate(args.tokenizers):
            subfolder = f"tokenizer_{idx + 1}" if idx > 0 else "tokenizer"
            tokenizers.append(Tokenizer.from_pretrained(tok_name, subfolder))
        model_config, topology_config, flagfile, tuning_spec, args = get_configs(args)
        model_params = ModelParams.load_json(model_config)
        vmfbs, params = get_modules(args, model_config, flagfile, tuning_spec)
        sdxl_service = GenerateService(
            name="sd",
            sysman=self.sysman,
            tokenizers=tokenizers,
            model_params=model_params,
            fibers_per_device=args.fibers_per_device,
            workers_per_device=args.workers_per_device,
            prog_isolation=args.isolation,
            show_progress=args.show_progress,
            trace_execution=args.trace_execution,
        )
        for key, vmfblist in vmfbs.items():
            for vmfb in vmfblist:
                sdxl_service.load_inference_module(vmfb, component=key)
        for key, datasets in params.items():
            sdxl_service.load_inference_parameters(*datasets, parameter_scope="model", component=key)
        sdxl_service.start()
        start = time.time()
        reps = args.reps
        procs = []
        for i in range(reps):
            service = MicroSDXLServer(args, sdxl_service)
            procs.append(service)
        await asyncio.gather(*[proc.launch() for proc in procs])
        print(f"Completed {reps} reps in {time.time() - start} seconds.")
        imgs = []
        for process in procs:
            imgs.append(process.imgs)

        return imgs

def run_cli(argv):
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
        help="Primary inferencing device",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        default="gfx942",
        choices=["gfx942", "gfx1100", "gfx90a"],
        help="Primary inferencing device LLVM target arch.",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        nargs="*",
        default=None,
        help="Device IDs visible to the system builder. Defaults to None (full visibility). Can be an index or a sf device id like amdgpu:0:0@0",
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
        choices=["compile", "precompiled"],
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
        "--prompt",
        type=str,
        default="a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal",
        help="Image generation prompt",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="Watermark, blurry, oversaturated, low resolution, pollution",
        help="Image generation negative prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default="20",
        help="Number of inference steps. More steps usually means a better image. Interactive only.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default="0.7",
        help="Guidance scale for denoising.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for image latents.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Benchmark samples.",
    )
    args = parser.parse_args(argv)
    if not args.artifacts_dir:
        home = Path.home()
        artdir = home / ".cache" / "shark"
        args.artifacts_dir = str(artdir)
    else:
        args.artifacts_dir = os.path.abspath(args.artifacts_dir)
    sysman = SystemManager(args.device, args.device_ids, args.amdgpu_async_allocations)
    main = Main(sysman)
    imgs = sysman.ls.run(main.main(args))
    print(f"number of images generated: {len(imgs)}")

if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)
    run_cli(
        sys.argv[1:],
    )
