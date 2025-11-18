# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import logging
import asyncio
from pathlib import Path
from typing import List
import numpy as np
import sys
import time
import os
import subprocess

# Import first as it does dep checking and reporting.
from shortfin.support.logging_setup import native_handler
import shortfin as sf


from .components.messages import InferencePhase, SDXLInferenceExecRequest
from .components.config_struct import ModelParams
from .components.manager import SystemManager
from .components.service import InferenceExecutorProcess, SDXLGenerateService
from .components.tokenizer import Tokenizer


logger = logging.getLogger("shortfin-sd")
logger.addHandler(native_handler)
logger.propagate = False

THIS_DIR = Path(__file__).parent


def get_configs(
    model_config,
    flagfile,
    target,
    artifacts_dir,
    use_tuned=True,
):
    # Returns one set of config artifacts.
    modelname = "sdxl"
    tuning_spec = None
    cfg_builder_args = [
        sys.executable,
        "-m",
        "iree.build",
        os.path.join(THIS_DIR, "components", "config_artifacts.py"),
        f"--target={target}",
        f"--output-dir={artifacts_dir}",
        f"--model={modelname}",
    ]
    outs = subprocess.check_output(cfg_builder_args).decode()
    outs_paths = outs.splitlines()
    for i in outs_paths:
        if "sdxl_config" in i and not model_config:
            model_config = i
        elif "flagfile" in i and not flagfile:
            flagfile = i
        elif "attention_and_matmul_spec" in i and use_tuned:
            tuning_spec = i

    if use_tuned and tuning_spec:
        tuning_spec = os.path.abspath(tuning_spec)

    return model_config, flagfile, tuning_spec


def get_modules(
    target,
    device,
    model_config,
    flagfile=None,
    td_spec=None,
    extra_compile_flags=[],
    artifacts_dir=None,
    splat=False,
    build_preference="export",
    force_update=False,
):
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
    model_flags["all"] = extra_compile_flags

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
            f"--target={target}",
            f"--splat={splat}",
            f"--build-preference={build_preference}",
            f"--output-dir={artifacts_dir}",
            f"--model={modelname}",
            f"--force-update={force_update}",
            f"--iree-hal-target-device={device}",
            f"--iree-hip-target={target}",
            f"--iree-compile-extra-args={ireec_extra_args}",
        ]
        logger.info(f"Preparing runtime artifacts for {modelname}...")
        logger.info(
            "COMMAND LINE EQUIVALENT: " + " ".join([str(argn) for argn in builder_args])
        )
        output = subprocess.check_output(builder_args, env=builder_env).decode()

        output_paths = output.splitlines()
        for path in output_paths:
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


class MicroSDXLExecutor(sf.Process):
    def __init__(self, args, service):
        super().__init__(fiber=service.meta_fibers[0].fiber)
        self.service = service

        self.args = args
        self.batch_size = args.batch_size
        self.exec = None
        self.imgs = None
        self.prompt = ""
        self.neg_prompt = args.neg_prompt
        self.steps = args.steps
        self.guidance_scale = args.guidance_scale

    async def run(self):
        # mostly its also defined as a method of the Process class probably envoked during launch call.

        # sample = [np.ones([1, 4, 128, 128], dtype=np.float16)] * self.batch_size

        self.exec = SDXLInferenceExecRequest(
            prompt=self.prompt,
            neg_prompt=self.neg_prompt,
            input_ids=None,
            height=self.args.output_image_resolution,
            width=self.args.output_image_resolution,
            steps=self.steps,
            guidance_scale=self.guidance_scale,
            sample=None,
        )

        self.exec.phases[InferencePhase.POSTPROCESS]["required"] = False
        while len(self.service.idle_meta_fibers) == 0:
            time.sleep(0.5)
            print("All fibers busy...")
        fiber = self.service.idle_meta_fibers.pop()
        exec_process = InferenceExecutorProcess(self.service, fiber)
        if self.service.prog_isolation != sf.ProgramIsolation.PER_FIBER:
            self.service.idle_meta_fibers.append(fiber)
        exec_process.exec_request = self.exec
        exec_process.launch()
        await asyncio.gather(exec_process)
        imgs = []
        await self.exec.done

        imgs.append(exec_process.exec_request.image_array)

        self.imgs = imgs
        return


class SDXLSampleProcessor:
    def __init__(self, service, prompt):
        self.service = service
        self.max_procs = 2
        self.num_procs = 0
        self.imgs = []
        self.procs = set()
        self.prompt = prompt

    def process(self, args):
        proc = MicroSDXLExecutor(args, self.service)
        proc.prompt = self.prompt
        self.num_procs += 1
        proc.launch()
        self.procs.add(proc)
        return

    def read(self):
        items = set()
        for proc in self.procs:
            if proc.imgs is not None:
                img = proc.imgs
                self.procs.remove(proc)
                self.num_procs -= 1
                return img
        return None


def create_service(
    model_params,
    device,
    tokenizers,
    vmfbs,
    params,
    device_idx=None,
    device_ids=[],
    fibers_per_device=1,
    isolation="per_call",
    trace_execution=False,
    amdgpu_async_allocations=False,
):
    if device_idx is not None:
        sysman = SystemManager(device, [device_idx], amdgpu_async_allocations)
    else:
        sysman = SystemManager(device, device_ids, amdgpu_async_allocations)

    sdxl_service = SDXLGenerateService(
        name="sd",
        sysman=sysman,
        tokenizers=tokenizers,
        model_params=model_params,
        fibers_per_device=fibers_per_device,
        workers_per_device=1,
        prog_isolation=isolation,
        show_progress=False,
        trace_execution=trace_execution,
    )
    for key, bs in vmfbs.items():
        for bs_int, vmfb_list in bs.items():
            for vmfb in vmfb_list:
                sdxl_service.load_inference_module(
                    vmfb, component=key, batch_size=bs_int
                )
    for key, datasets in params.items():
        sdxl_service.load_inference_parameters(
            *datasets, parameter_scope="model", component=key
        )
    sdxl_service.start()
    return sdxl_service


def prepare_service(args):
    tokenizers = []
    for idx, tok_name in enumerate(args.tokenizers):
        subfolder = f"tokenizer_{idx + 1}" if idx > 0 else "tokenizer"
        tokenizers.append(Tokenizer.from_pretrained(tok_name, subfolder))
    model_config, flagfile, tuning_spec = get_configs(
        args.model_config,
        args.flagfile,
        args.target,
        args.artifacts_dir,
        args.use_tuned,
    )
    model_params = ModelParams.load_json(model_config)
    vmfbs, params = get_modules(
        args.target,
        args.device,
        model_config,
        flagfile,
        tuning_spec,
        artifacts_dir=args.artifacts_dir,
        build_preference=args.build_preference,
    )
    return model_params, tokenizers, vmfbs, params


async def run_benchmark(
    sysman,
    args,
    input_token_length,
):
    """Execute the benchmark and return raw data."""
    model_params, tokenizers, vmfbs, params = prepare_service(args)
    services = set()

    prompt = " ".join(["one" for _ in range(input_token_length)])

    for idx, _ in enumerate(sysman.ls.device_names):
        services.add(
            create_service(
                model_params,
                args.device,
                tokenizers,
                vmfbs,
                params,
                device_idx=idx,
                trace_execution=args.trace_execution,
                amdgpu_async_allocations=args.amdgpu_async_allocations,
            )
        )

    procs = set()
    procs_per_service = 2
    for service in services:
        for i in range(procs_per_service):
            sample_processor = SDXLSampleProcessor(service, prompt)
            procs.add(sample_processor)

    samples = args.samples
    queue = set()
    # n sets of arguments into a queue

    for i in range(samples):
        # Run until told to stop or queue exhaustion
        # OR multiple dequeue threads pulling from queue
        # read, instantiate, launch
        # knob : concurrency control
        queue.add(i)

    start = time.time()
    imgs = []
    # Fire off jobs
    while len(queue) > 0:
        # round robin pop items from queue into executors
        this_processor = procs.pop()
        while this_processor.num_procs >= this_processor.max_procs:
            procs.add(this_processor)
            this_processor = procs.pop()
            # Try reading and clearing out processes before checking again.
            for proc in procs:
                results = proc.read()
                if results:
                    imgs.append(results)
                    print(f"{len(imgs)} samples received, of a total {samples}")
        # Pop item from queue and initiate process.
        queue.pop()
        this_processor.process(args)
        procs.add(this_processor)

    # Read responses
    while len(imgs) < samples:
        for proc in procs:
            results = proc.read()
            if results:
                imgs.append(results)
                print(f"{len(imgs)} samples received, of a total {samples}")

    print(f"Completed {samples} samples in {time.time() - start} seconds.")
    return


async def run_all_benchmarks(sysman, args):

    input_token_lengths = args.input_token_lengths
    output_image_resolution = args.output_image_resolution

    for input_token_length in input_token_lengths:

        print(
            f"\n\nRunning benchmarks with input_token_length = {input_token_length} and output image resolution = {output_image_resolution}"
        )

        benchmark_data = await run_benchmark(
            sysman=sysman,
            args=args,
            input_token_length=int(input_token_length),
        )


def main():
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
        choices=["gfx90a", "gfx942", "gfx1100", "gfx1201"],
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
        "--input_token_lengths",
        type=int,
        nargs="+",
        default="32",
        help="Image generation input token length",
    )
    parser.add_argument(
        "--output_image_resolution",
        type=int,
        nargs="+",
        default="1024",
        help="Image generation output resolution",
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
        "--samples",
        type=int,
        default=1,
        help="Benchmark samples.",
    )
    parser.add_argument(
        "--max_concurrent_procs",
        type=int,
        default=16,
        help="Maximum number of executor threads to run at any given time.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference program batch size to test.",
    )
    args = parser.parse_args()
    if not args.artifacts_dir:
        home = Path.home()
        artdir = home / ".cache" / "amdshark"
        args.artifacts_dir = str(artdir)
    else:
        args.artifacts_dir = os.path.abspath(args.artifacts_dir)

    sysman = SystemManager(args.device, args.device_ids, args.amdgpu_async_allocations)

    asyncio.run(run_all_benchmarks(sysman, args))


if __name__ == "__main__":
    main()
