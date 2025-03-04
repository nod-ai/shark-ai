# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import math
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Type
from PIL import Image
import base64

import shortfin as sf
import shortfin.array as sfnp

from ...utils import ServiceBase, BatcherProcessBase, prog_isolations

from .config_struct import ModelParams
from .manager import SystemManager
from .messages import FluxInferenceExecRequest, InferencePhase
from .tokenizer import Tokenizer
from .metrics import measure

logger = logging.getLogger("shortfin-flux.service")


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


class GenerateService(ServiceBase):
    """Top level service interface for image generation."""

    inference_programs: dict[str, sf.Program]

    inference_functions: dict[str, dict[str, sf.ProgramFunction]]

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        clip_tokenizers: list[Tokenizer],
        t5xxl_tokenizers: list[Tokenizer],
        model_params: ModelParams,
        fibers_per_device: int,
        workers_per_device: int = 1,
        prog_isolation: str = "per_fiber",
        show_progress: bool = False,
        trace_execution: bool = False,
    ):
        super().__init__(sysman)
        self.name = name

        # Application objects.
        self.clip_tokenizers = clip_tokenizers
        self.t5xxl_tokenizers = t5xxl_tokenizers
        self.model_params = model_params
        self.inference_functions: dict[str, dict[str, sf.ProgramFunction]] = {}
        self.inference_programs: dict[int, dict[str, sf.Program]] = {}
        self.trace_execution = trace_execution
        self.show_progress = show_progress

        self.prog_isolation = prog_isolations[prog_isolation]

        self.workers_per_device = workers_per_device
        self.fibers_per_device = fibers_per_device
        if fibers_per_device % workers_per_device != 0:
            raise ValueError(
                "Currently, fibers_per_device must be divisible by workers_per_device"
            )
        self.fibers_per_worker = int(fibers_per_device / workers_per_device)

        self.workers = []
        self.fibers = []
        self.idle_fibers = set()
        # For each worker index we create one on each device, and add their fibers to the idle set.
        # This roughly ensures that the first picked fibers are distributed across available devices.
        for i in range(self.workers_per_device):
            for idx, device in enumerate(self.sysman.ls.devices):
                worker = sysman.ls.create_worker(f"{name}-inference-{device.name}-{i}")
                self.workers.append(worker)
        for idx, device in enumerate(self.sysman.ls.devices):
            for i in range(self.fibers_per_device):
                tgt_worker = self.workers[i % len(self.workers)]
                fiber = sysman.ls.create_fiber(tgt_worker, devices=[device])
                self.fibers.append(fiber)
                self.idle_fibers.add(fiber)
        for idx in range(len(self.workers)):
            self.inference_programs[idx] = {}
            self.inference_functions[idx] = {
                "clip": {},
                "t5xxl": {},
                "denoise": {},
                "decode": {},
            }
        # Scope dependent objects.
        self.batcher = FluxBatcherProcess(self)

    def get_worker_index(self, fiber):
        if fiber not in self.fibers:
            raise ValueError("A worker was requested from a rogue fiber.")
        fiber_idx = self.fibers.index(fiber)
        worker_idx = int(
            (fiber_idx - fiber_idx % self.fibers_per_worker) / self.fibers_per_worker
        )
        return worker_idx

    def start(self):
        # Initialize programs.
        for component in self.inference_modules:
            logger.info(f"Loading component: {component}")
            component_modules = [
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters.get(component, [])
                ),
                *self.inference_modules[component],
            ]

            for worker_idx, worker in enumerate(self.workers):
                worker_devices = self.fibers[
                    worker_idx * (self.fibers_per_worker)
                ].raw_devices
                logger.info(
                    f"Loading inference program: {component}, worker index: {worker_idx}, device: {worker_devices}"
                )
                self.inference_programs[worker_idx][component] = sf.Program(
                    modules=component_modules,
                    devices=worker_devices,
                    isolation=self.prog_isolation,
                    trace_execution=self.trace_execution,
                )

        for worker_idx, worker in enumerate(self.workers):
            for bs in self.model_params.clip_batch_sizes:
                self.inference_functions[worker_idx]["clip"][
                    bs
                ] = self.inference_programs[worker_idx]["clip"][
                    f"{self.model_params.clip_module_name}.encode_prompts"
                ]
            for bs in self.model_params.t5xxl_batch_sizes:
                self.inference_functions[worker_idx]["t5xxl"][
                    bs
                ] = self.inference_programs[worker_idx]["t5xxl"][
                    f"{self.model_params.t5xxl_module_name}.encode_prompts"
                ]
            self.inference_functions[worker_idx]["denoise"] = {}
            for bs in self.model_params.sampler_batch_sizes:
                self.inference_functions[worker_idx]["denoise"][bs] = {
                    "sampler": self.inference_programs[worker_idx]["sampler"][
                        f"{self.model_params.sampler_module_name}.{self.model_params.sampler_fn_name}"
                    ],
                }
            self.inference_functions[worker_idx]["decode"] = {}
            for bs in self.model_params.vae_batch_sizes:
                self.inference_functions[worker_idx]["decode"][
                    bs
                ] = self.inference_programs[worker_idx]["vae"][
                    f"{self.model_params.vae_module_name}.decode"
                ]
        self.batcher.launch()

    def shutdown(self):
        self.batcher.shutdown()

    def __repr__(self):
        modules = [
            f"     {key} : {value}" for key, value in self.inference_modules.items()
        ]
        params = [
            f"     {key} : {value}" for key, value in self.inference_parameters.items()
        ]
        # For python 3.11 since we can't have \ in the f"" expression.
        new_line = "\n"
        return (
            f"ServiceManager("
            f"\n  INFERENCE DEVICES : \n"
            f"     {self.sysman.ls.devices}\n"
            f"\n  MODEL PARAMS : \n"
            f"{self.model_params}"
            f"\n  SERVICE PARAMS : \n"
            f"     fibers per device : {self.fibers_per_device}\n"
            f"     program isolation mode : {self.prog_isolation}\n"
            f"\n  INFERENCE MODULES : \n"
            f"{new_line.join(modules)}\n"
            f"\n  INFERENCE PARAMETERS : \n"
            f"{new_line.join(params)}\n"
            f")"
        )


########################################################################################
# Sequential Service (Memory-Efficient)
########################################################################################


class SequentialGenerateService(GenerateService):
    """A memory-efficient version of GenerateService that loads models sequentially."""

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        clip_tokenizers: list[Tokenizer],
        t5xxl_tokenizers: list[Tokenizer],
        model_params: ModelParams,
        fibers_per_device: int,
        workers_per_device: int = 1,
        prog_isolation: str = "per_fiber",
        show_progress: bool = False,
        trace_execution: bool = False,
        split_denoise: bool = False,
    ):
        super().__init__(
            name=name,
            sysman=sysman,
            clip_tokenizers=clip_tokenizers,
            t5xxl_tokenizers=t5xxl_tokenizers,
            model_params=model_params,
            fibers_per_device=fibers_per_device,
            workers_per_device=workers_per_device,
            prog_isolation=prog_isolation,
            show_progress=show_progress,
            trace_execution=trace_execution,
        )
        self.batcher = SequentialFluxBatcherProcess(self)
        self.loaded_components: Set[str] = set()
        self.model_lock = asyncio.Lock()
        self.split_denoise = split_denoise
        
        for idx in range(len(self.workers)):
            if self.split_denoise:
                self.inference_functions[idx]["denoise_front"] = {}
                self.inference_functions[idx]["denoise_back"] = {}

    async def load_component(self, component_name: str, worker_idx: int):
        async with self.model_lock:
            if component_name in self.loaded_components:
                return

            logger.info(f"Loading component: {component_name}")
            
            # For split denoise, we need to handle front/back specially
            if self.split_denoise and component_name in ["sampler_front", "sampler_back"]:
                if "sampler_front" in self.loaded_components and component_name == "sampler_back":
                    await self.unload_component("sampler_front", worker_idx)
                elif "sampler_back" in self.loaded_components and component_name == "sampler_front":
                    await self.unload_component("sampler_back", worker_idx)
            else:
                # For other components, unload everything
                await self.unload_all_components(worker_idx)
            
            # Map the front/back components to the right modules
            actual_component = component_name
            if component_name in ["sampler_front", "sampler_back"]:
                actual_component = "sampler"
            
            component_modules = [
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters.get(actual_component, [])
                ),
                *self.inference_modules[actual_component],
            ]

            worker_devices = self.fibers[
                worker_idx * (self.fibers_per_worker)
            ].raw_devices
            
            logger.info(
                f"Loading inference program: {component_name}, worker index: {worker_idx}, device: {worker_devices}"
            )
            
            self.inference_programs[worker_idx][component_name] = sf.Program(
                modules=component_modules,
                devices=worker_devices,
                isolation=self.prog_isolation,
                trace_execution=self.trace_execution,
            )
            
            if component_name == "clip":
                for bs in self.model_params.clip_batch_sizes:
                    self.inference_functions[worker_idx]["clip"][bs] = self.inference_programs[worker_idx]["clip"][
                        f"{self.model_params.clip_module_name}.encode_prompts"
                    ]
            elif component_name == "t5xxl":
                for bs in self.model_params.t5xxl_batch_sizes:
                    self.inference_functions[worker_idx]["t5xxl"][bs] = self.inference_programs[worker_idx]["t5xxl"][
                        f"{self.model_params.t5xxl_module_name}.encode_prompts"
                    ]
            elif component_name == "sampler":
                self.inference_functions[worker_idx]["denoise"] = {}
                for bs in self.model_params.sampler_batch_sizes:
                    self.inference_functions[worker_idx]["denoise"][bs] = {
                        "sampler": self.inference_programs[worker_idx]["sampler"][
                            f"{self.model_params.sampler_module_name}.{self.model_params.sampler_fn_name}"
                        ],
                    }
            elif component_name == "sampler_front":
                self.inference_functions[worker_idx]["denoise_front"] = {}
                for bs in self.model_params.sampler_batch_sizes:
                    self.inference_functions[worker_idx]["denoise_front"][bs] = {
                        "sampler_front": self.inference_programs[worker_idx]["sampler_front"][
                            f"{self.model_params.sampler_module_name}.{self.model_params.sampler_fn_name}_front"
                        ],
                    }
            elif component_name == "sampler_back":
                self.inference_functions[worker_idx]["denoise_back"] = {}
                for bs in self.model_params.sampler_batch_sizes:
                    self.inference_functions[worker_idx]["denoise_back"][bs] = {
                        "sampler_back": self.inference_programs[worker_idx]["sampler_back"][
                            f"{self.model_params.sampler_module_name}.{self.model_params.sampler_fn_name}_back"
                        ],
                    }
            elif component_name == "vae":
                self.inference_functions[worker_idx]["decode"] = {}
                for bs in self.model_params.vae_batch_sizes:
                    self.inference_functions[worker_idx]["decode"][bs] = self.inference_programs[worker_idx]["vae"][
                        f"{self.model_params.vae_module_name}.decode"
                    ]
            
            self.loaded_components.add(component_name)

    async def unload_component(self, component_name: str, worker_idx: int):
        async with self.model_lock:
            if component_name not in self.loaded_components:
                return
            
            logger.info(f"Unloading component: {component_name}")
            
            if component_name in self.inference_programs[worker_idx]:
                del self.inference_programs[worker_idx][component_name]
                
            if component_name == "clip":
                self.inference_functions[worker_idx]["clip"] = {}
            elif component_name == "t5xxl":
                self.inference_functions[worker_idx]["t5xxl"] = {}
            elif component_name == "sampler":
                self.inference_functions[worker_idx]["denoise"] = {}
            elif component_name == "sampler_front":
                self.inference_functions[worker_idx]["denoise_front"] = {}
            elif component_name == "sampler_back":
                self.inference_functions[worker_idx]["denoise_back"] = {}
            elif component_name == "vae":
                self.inference_functions[worker_idx]["decode"] = {}
            
            self.loaded_components.remove(component_name)
            
            import gc
            gc.collect()

    async def unload_all_components(self, worker_idx: int):
        components_to_unload = list(self.loaded_components)
        for component in components_to_unload:
            await self.unload_component(component, worker_idx)

    def start(self):
        self.batcher.launch()

    def get_component_for_phase(self, phase: InferencePhase) -> list[str]:
        if self.split_denoise:
            phase_to_component_map = {
                InferencePhase.ENCODE: ["clip", "t5xxl"],
                InferencePhase.DENOISE: ["sampler_front", "sampler_back"],
                InferencePhase.DECODE: ["vae"],
            }
        else:
            phase_to_component_map = {
                InferencePhase.ENCODE: ["clip", "t5xxl"],
                InferencePhase.DENOISE: ["sampler"],
                InferencePhase.DECODE: ["vae"],
            }
        
        if phase in phase_to_component_map:
            return phase_to_component_map[phase]
        return []


########################################################################################
# Batcher
########################################################################################


class FluxBatcherProcess(BatcherProcessBase):
    STROBE_SHORT_DELAY = 0.5
    STROBE_LONG_DELAY = 1

    def __init__(self, service: GenerateService):
        super().__init__(fiber=service.fibers[0])
        self.service = service
        self.ideal_batch_size: int = max(service.model_params.max_batch_size)
        self.num_fibers = len(service.fibers)

    def handle_inference_request(self, request):
        self.pending_requests.add(request)

    async def process_batches(self):
        await self.board_flights()

    async def board_flights(self):
        waiting_count = len(self.pending_requests)
        if waiting_count == 0:
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return
        self.strobes = 0
        batches = self.sort_batches()
        for batch in batches.values():
            # Assign the batch to the next idle fiber.
            if len(self.service.idle_fibers) == 0:
                return
            fiber = self.service.idle_fibers.pop()
            fiber_idx = self.service.fibers.index(fiber)
            worker_idx = self.service.get_worker_index(fiber)
            logger.debug(f"Sending batch to fiber {fiber_idx} (worker {worker_idx})")
            self.board(batch["reqs"], fiber=fiber)
            if self.service.prog_isolation != sf.ProgramIsolation.PER_FIBER:
                self.service.idle_fibers.add(fiber)

    def board(self, request_bundle, fiber):
        pending = request_bundle
        if len(pending) == 0:
            return
        exec_process = InferenceExecutorProcess(self.service, fiber)
        for req in pending:
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            exec_process.exec_requests.append(req)
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending_requests.remove(flighted_request)
            exec_process.launch()


class SequentialFluxBatcherProcess(FluxBatcherProcess):
    """Batcher process adapted for sequential model loading/unloading."""

    def board(self, request_bundle, fiber):
        pending = request_bundle
        if len(pending) == 0:
            return
        exec_process = SequentialInferenceExecutorProcess(self.service, fiber)
        for req in pending:
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            exec_process.exec_requests.append(req)
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending_requests.remove(flighted_request)
            exec_process.launch()


########################################################################################
# Inference Executors
########################################################################################


class InferenceExecutorProcess(sf.Process):
    """Executes a stable diffusion inference batch"""

    def __init__(
        self,
        service: GenerateService,
        fiber,
    ):
        super().__init__(fiber=fiber)
        self.service = service
        self.worker_index = self.service.get_worker_index(fiber)
        self.exec_requests: list[FluxInferenceExecRequest] = []

    @measure(type="exec", task="inference process")
    async def run(self):
        try:
            phase = None
            for req in self.exec_requests:
                if phase:
                    if phase != req.phase:
                        logger.error("Executor process recieved disjoint batch.")
                phase = req.phase
            phases = self.exec_requests[0].phases
            req_count = len(self.exec_requests)
            device0 = self.fiber.device(0)
            if phases[InferencePhase.PREPARE]["required"]:
                await self._prepare(device=device0, requests=self.exec_requests)
            if phases[InferencePhase.ENCODE]["required"]:
                await self._clip(device=device0, requests=self.exec_requests)
                await self._t5xxl(device=device0, requests=self.exec_requests)
            if phases[InferencePhase.DENOISE]["required"]:
                await self._denoise(device=device0, requests=self.exec_requests)
            if phases[InferencePhase.DECODE]["required"]:
                await self._decode(device=device0, requests=self.exec_requests)
            if phases[InferencePhase.POSTPROCESS]["required"]:
                await self._postprocess(device=device0, requests=self.exec_requests)
            await device0
            for i in range(req_count):
                req = self.exec_requests[i]
                req.done.set_success()
            if self.service.prog_isolation == sf.ProgramIsolation.PER_FIBER:
                self.service.idle_fibers.add(self.fiber)

        except Exception:
            logger.exception("Fatal error in image generation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.done.set_success()

    async def _prepare(self, device, requests):
        for request in requests:
            # Tokenize prompts and negative prompts. We tokenize in bs1 for now and join later.
            clip_input_ids_list = []
            clip_neg_ids_list = []
            for tokenizer in self.service.clip_tokenizers:
                input_ids = tokenizer.encode(request.prompt)
                clip_input_ids_list.append(input_ids)
                neg_ids = tokenizer.encode(request.neg_prompt)
                clip_neg_ids_list.append(neg_ids)
            clip_ids_list = [*clip_input_ids_list, *clip_neg_ids_list]

            request.clip_input_ids = clip_ids_list

            t5xxl_input_ids_list = []
            t5xxl_neg_ids_list = []
            for tokenizer in self.service.t5xxl_tokenizers:
                input_ids = tokenizer.encode(request.prompt)
                t5xxl_input_ids_list.append(input_ids)
                neg_ids = tokenizer.encode(request.neg_prompt)
                t5xxl_neg_ids_list.append(neg_ids)
            t5xxl_ids_list = [*t5xxl_input_ids_list, *t5xxl_neg_ids_list]

            request.t5xxl_input_ids = t5xxl_ids_list

            # Generate random sample latents.
            seed = request.seed
            channels = self.service.model_params.num_latents_channels
            image_seq_len = (request.height) * (request.width) // 256
            latents_shape = [
                1,
                image_seq_len,
                64,
            ]

            # Create and populate sample device array.
            generator = sfnp.RandomGenerator(seed)
            request.sample = sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.sampler_dtype
            )

            sample_host = sfnp.device_array.for_host(
                device, latents_shape, sfnp.float32
            )
            with sample_host.map(discard=True) as m:
                m.fill(bytes(1))
            sfnp.fill_randn(sample_host, generator=generator)
            if self.service.model_params.sampler_dtype != sfnp.float32:
                sample_transfer = request.sample.for_transfer()
                sfnp.convert(
                    sample_host,
                    dtype=self.service.model_params.sampler_dtype,
                    out=sample_transfer,
                )

                request.sample.copy_from(sample_transfer)
            else:
                request.sample.copy_from(sample_host)

            await device
            request.timesteps = get_schedule(
                request.steps,
                image_seq_len,
                shift=not self.service.model_params.is_schnell,
            )
        return

    async def _clip(self, device, requests):
        req_bs = len(requests)
        entrypoints = self.service.inference_functions[self.worker_index]["clip"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._clip(device, [request])
            return
        for bs, fn in entrypoints.items():
            if bs == req_bs:
                break

        # Prepare tokenized input ids for CLIP inference
        clip_inputs = [
            sfnp.device_array.for_device(
                device,
                [req_bs, self.service.model_params.clip_max_seq_len],
                sfnp.sint64,
            ),
        ]
        host_arrs = [None]
        for idx, arr in enumerate(clip_inputs):
            host_arrs[idx] = arr.for_transfer()
            for i in range(req_bs):
                with host_arrs[idx].view(i).map(write=True, discard=True) as m:

                    num_ids = len(requests[i].clip_input_ids)
                    np_arr = requests[i].clip_input_ids[idx % (num_ids - 1)].input_ids

                    m.fill(np_arr)
            clip_inputs[idx].copy_from(host_arrs[idx])

        # Encode tokenized inputs.
        logger.debug(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(clip_inputs)]),
        )
        await device
        (vec,) = await fn(*clip_inputs, fiber=self.fiber)
        await device

        for i in range(req_bs):
            cfg_mult = 1
            requests[i].vec = vec.view(slice(i, (i + 1)))

        a = vec.for_transfer()
        a.copy_from(vec)
        await device

        return

    async def _t5xxl(self, device, requests):
        req_bs = len(requests)
        entrypoints = self.service.inference_functions[self.worker_index]["t5xxl"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._t5xxl(device, [request])
            return
        for bs, fn in entrypoints.items():
            if bs == req_bs:
                break

        # Prepare tokenized input ids for t5xxl inference
        t5xxl_inputs = [
            sfnp.device_array.for_device(
                device, [1, self.service.model_params.t5xxl_max_seq_len], sfnp.sint64
            ),
        ]
        host_arrs = [None]
        for idx, arr in enumerate(t5xxl_inputs):
            host_arrs[idx] = arr.for_transfer()
            for i in range(req_bs):
                np_arr = requests[i].t5xxl_input_ids[idx].input_ids
                with host_arrs[idx].view(0).map(write=True, discard=True) as m:
                    m.fill(np_arr)
            t5xxl_inputs[idx].copy_from(host_arrs[idx])

        # Encode tokenized inputs.
        logger.debug(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(t5xxl_inputs)]),
        )
        await device
        (txt,) = await fn(*t5xxl_inputs, fiber=self.fiber)
        await device
        for i in range(req_bs):
            cfg_mult = requests[i].cfg_mult
            requests[i].txt = txt.view(slice(i * cfg_mult, (i + 1) * cfg_mult))

        return

    async def _denoise(self, device, requests):
        req_bs = len(requests)
        step_count = requests[0].steps
        print(step_count)
        cfg_mult = requests[0].cfg_mult

        # Produce denoised latents
        entrypoints = self.service.inference_functions[self.worker_index]["denoise"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._denoise(device, [request])
            return
        for bs, fns in entrypoints.items():
            if bs == req_bs:
                break

        # Get shape of batched latents.
        # This assumes all requests are dense at this point.
        img_shape = [
            req_bs * cfg_mult,
            (requests[0].height) * (requests[0].width) // 256,
            64,
        ]
        # Assume we are doing classifier-free guidance
        txt_shape = [
            req_bs * cfg_mult,
            self.service.model_params.t5xxl_max_seq_len,
            self.service.model_params.t5xxl_out_dim,
        ]
        vec_shape = [
            req_bs * cfg_mult,
            self.service.model_params.clip_out_dim,
        ]
        denoise_inputs = {
            "img": sfnp.device_array.for_device(
                device, img_shape, self.service.model_params.sampler_dtype
            ),
            "txt": sfnp.device_array.for_device(
                device, txt_shape, self.service.model_params.sampler_dtype
            ),
            "vec": sfnp.device_array.for_device(
                device, vec_shape, self.service.model_params.sampler_dtype
            ),
            "step": sfnp.device_array.for_device(device, [1], sfnp.int64),
            "timesteps": sfnp.device_array.for_device(
                device, [100], self.service.model_params.sampler_dtype
            ),
            "guidance_scale": sfnp.device_array.for_device(
                device, [req_bs], self.service.model_params.sampler_dtype
            ),
        }
        # Send guidance scale to device.
        gs_host = denoise_inputs["guidance_scale"].for_transfer()
        sample_host = sfnp.device_array.for_host(
            device, img_shape, self.service.model_params.sampler_dtype
        )
        guidance_float = sfnp.device_array.for_host(device, [req_bs], sfnp.float32)

        for i in range(req_bs):
            guidance_float.view(i).items = [requests[i].guidance_scale]
            cfg_dim = i * cfg_mult

            # Reshape and batch sample latent inputs on device.
            # Currently we just generate random latents in the desired shape. Rework for img2img.
            req_samp = requests[i].sample
            for rep in range(cfg_mult):
                sample_host.view(slice(cfg_dim + rep, cfg_dim + rep + 1)).copy_from(
                    req_samp
                )
            denoise_inputs["img"].view(slice(cfg_dim, cfg_dim + cfg_mult)).copy_from(
                sample_host
            )

            # Batch t5xxl hidden states.
            txt = requests[i].txt
            if (
                self.service.model_params.t5xxl_dtype
                != self.service.model_params.sampler_dtype
            ):
                inter = sfnp.device_array.for_host(
                    device, txt_shape, dtype=self.service.model_params.sampler_dtype
                )
                host = sfnp.device_array.for_host(
                    device, txt_shape, dtype=self.service.model_params.t5xxl_dtype
                )
                host.view(slice(cfg_dim, cfg_dim + cfg_mult)).copy_from(txt)
                await device
                sfnp.convert(
                    host,
                    dtype=self.service.model_params.sampler_dtype,
                    out=inter,
                )
                denoise_inputs["txt"].view(
                    slice(cfg_dim, cfg_dim + cfg_mult)
                ).copy_from(inter)
            else:
                denoise_inputs["txt"].view(
                    slice(cfg_dim, cfg_dim + cfg_mult)
                ).copy_from(txt)

            # Batch CLIP projections.
            vec = requests[i].vec
            if (
                self.service.model_params.t5xxl_dtype
                != self.service.model_params.sampler_dtype
            ):
                for nc in range(cfg_mult):
                    inter = sfnp.device_array.for_host(
                        device, vec_shape, dtype=self.service.model_params.sampler_dtype
                    )
                    host = sfnp.device_array.for_host(
                        device, vec_shape, dtype=self.service.model_params.clip_dtype
                    )
                    host.view(slice(nc, nc + 1)).copy_from(vec)
                    await device
                    sfnp.convert(
                        host,
                        dtype=self.service.model_params.sampler_dtype,
                        out=inter,
                    )
                    denoise_inputs["vec"].view(slice(nc, nc + 1)).copy_from(inter)
            else:
                for nc in range(cfg_mult):
                    denoise_inputs["vec"].view(slice(nc, nc + 1)).copy_from(vec)
        sfnp.convert(
            guidance_float, dtype=self.service.model_params.sampler_dtype, out=gs_host
        )
        denoise_inputs["guidance_scale"].copy_from(gs_host)
        await device
        ts_host = denoise_inputs["timesteps"].for_transfer()
        ts_float = sfnp.device_array.for_host(
            device, denoise_inputs["timesteps"].shape, dtype=sfnp.float32
        )
        with ts_float.map(write=True) as m:
            m.fill(float(1))
        for tstep in range(len(requests[0].timesteps)):
            with ts_float.view(tstep).map(write=True, discard=True) as m:
                m.fill(np.asarray(requests[0].timesteps[tstep], dtype="float32"))

        sfnp.convert(
            ts_float, dtype=self.service.model_params.sampler_dtype, out=ts_host
        )
        denoise_inputs["timesteps"].copy_from(ts_host)
        await device

        for i, t in tqdm(
            enumerate(range(step_count)),
            disable=(not self.service.show_progress),
            desc=f"DENOISE (bs{req_bs})",
        ):
            s_host = denoise_inputs["step"].for_transfer()
            with s_host.map(write=True) as m:
                s_host.items = [i]
            denoise_inputs["step"].copy_from(s_host)

            logger.info(
                "INVOKE %r",
                fns["sampler"],
            )
            await device
            (noise_pred,) = await fns["sampler"](
                *denoise_inputs.values(), fiber=self.fiber
            )
            await device
            denoise_inputs["img"].copy_from(noise_pred)

        for idx, req in enumerate(requests):
            req.denoised_latents = sfnp.device_array.for_device(
                device, img_shape, self.service.model_params.vae_dtype
            )
            if (
                self.service.model_params.vae_dtype
                != self.service.model_params.sampler_dtype
            ):
                pred_shape = [
                    1,
                    (requests[0].height) * (requests[0].width) // 256,
                    64,
                ]
                denoised_inter = sfnp.device_array.for_host(
                    device, pred_shape, dtype=self.service.model_params.vae_dtype
                )
                denoised_host = sfnp.device_array.for_host(
                    device, pred_shape, dtype=self.service.model_params.sampler_dtype
                )
                denoised_host.copy_from(denoise_inputs["img"].view(idx * cfg_mult))
                await device
                sfnp.convert(
                    denoised_host,
                    dtype=self.service.model_params.vae_dtype,
                    out=denoised_inter,
                )
                req.denoised_latents.copy_from(denoised_inter)
            else:
                req.denoised_latents.copy_from(
                    denoise_inputs["img"].view(idx * cfg_mult)
                )
        return

    async def _decode(self, device, requests):
        req_bs = len(requests)
        # Decode latents to images
        entrypoints = self.service.inference_functions[self.worker_index]["decode"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._decode(device, [request])
            return
        for bs, fn in entrypoints.items():
            if bs == req_bs:
                break
        latents_shape = [
            req_bs,
            (requests[0].height * requests[0].width) // 256,
            64,
        ]
        latents = sfnp.device_array.for_device(
            device, latents_shape, self.service.model_params.vae_dtype
        )

        for i in range(req_bs):
            latents.view(i).copy_from(requests[i].denoised_latents)

        # Decode the denoised latents.
        logger.debug(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  0: {latents.shape}"]),
        )
        await device
        (image,) = await fn(latents, fiber=self.fiber)
        await device
        images_shape = [
            req_bs,
            3,
            requests[0].height,
            requests[0].width,
        ]
        images_host = sfnp.device_array.for_host(
            device, images_shape, self.service.model_params.vae_dtype
        )
        images_host.copy_from(image)
        await device
        for idx, req in enumerate(requests):
            req.image_array = images_host.view(idx)
        return

    async def _postprocess(self, device, requests):
        # Process output images
        for req in requests:
            image_shape = [
                3,
                req.height,
                req.width,
            ]
            out_shape = [req.height, req.width, 3]
            images_planar = sfnp.device_array.for_host(
                device, image_shape, self.service.model_params.vae_dtype
            )
            images_planar.copy_from(req.image_array)
            await device
            permuted = sfnp.device_array.for_host(
                device, out_shape, self.service.model_params.vae_dtype
            )
            out = sfnp.device_array.for_host(device, out_shape, sfnp.uint8)
            sfnp.transpose(images_planar, (1, 2, 0), out=permuted)
            permuted = sfnp.multiply(127.5, (sfnp.add(permuted, 1.0)))
            out = sfnp.round(permuted, dtype=sfnp.uint8)
            image_bytes = bytes(out.map(read=True))

            image = base64.b64encode(image_bytes).decode("utf-8")
            req.result_image = image
        return


class SequentialInferenceExecutorProcess(InferenceExecutorProcess):
    """Executes inference with sequential model loading/unloading to conserve memory."""
    
    def __init__(
        self,
        service: SequentialGenerateService,
        fiber,
    ):
        super().__init__(service, fiber)
        self.sequential_service = service
    
    @measure(type="exec", task="sequential inference process")
    async def run(self):
        try:
            phase = None
            for req in self.exec_requests:
                if phase and phase != req.phase:
                    logger.error("Executor process received disjoint batch.")
                phase = req.phase
            phases = self.exec_requests[0].phases
            req_count = len(self.exec_requests)
            device0 = self.fiber.device(0)
            
            # Process each phase sequentially with model loading/unloading between phases
            if phases[InferencePhase.PREPARE]["required"]:
                await self._prepare(device=device0, requests=self.exec_requests)
                
            if phases[InferencePhase.ENCODE]["required"]:
                # Load CLIP model
                await self.sequential_service.load_component("clip", self.worker_index)
                await self._clip(device=device0, requests=self.exec_requests)
                # Unload CLIP to free memory before loading T5XXL
                await self.sequential_service.unload_component("clip", self.worker_index)
                
                # Load T5XXL model
                await self.sequential_service.load_component("t5xxl", self.worker_index)
                await self._t5xxl(device=device0, requests=self.exec_requests)
                # Unload T5XXL after using it
                await self.sequential_service.unload_component("t5xxl", self.worker_index)
                
            if phases[InferencePhase.DENOISE]["required"]:
                if self.sequential_service.split_denoise:
                    await self._split_denoise(device=device0, requests=self.exec_requests)
                else:
                    # Load regular sampler model
                    await self.sequential_service.load_component("sampler", self.worker_index)
                    await self._denoise(device=device0, requests=self.exec_requests)
                    # Unload sampler after using it
                    await self.sequential_service.unload_component("sampler", self.worker_index)
                
            if phases[InferencePhase.DECODE]["required"]:
                # Load VAE model
                await self.sequential_service.load_component("vae", self.worker_index)
                await self._decode(device=device0, requests=self.exec_requests)
                # Unload VAE after using it
                await self.sequential_service.unload_component("vae", self.worker_index)
                
            if phases[InferencePhase.POSTPROCESS]["required"]:
                await self._postprocess(device=device0, requests=self.exec_requests)
                
            await device0
            for i in range(req_count):
                req = self.exec_requests[i]
                req.done.set_success()
                
            if self.service.prog_isolation == sf.ProgramIsolation.PER_FIBER:
                self.service.idle_fibers.add(self.fiber)

        except Exception:
            logger.exception("Fatal error in sequential image generation")
            await self.sequential_service.unload_all_components(self.worker_index)
            for req in self.exec_requests:
                req.done.set_success()
    
    async def _split_denoise(self, device, requests):
        """Split denoise implementation that alternates between front and back halves."""
        req_bs = len(requests)
        step_count = requests[0].steps
        cfg_mult = requests[0].cfg_mult

        # Prepare all the same inputs as the regular denoise function
        img_shape = [
            req_bs * cfg_mult,
            (requests[0].height) * (requests[0].width) // 256,
            64,
        ]
        txt_shape = [
            req_bs * cfg_mult,
            self.service.model_params.t5xxl_max_seq_len,
            self.service.model_params.t5xxl_out_dim,
        ]
        vec_shape = [
            req_bs * cfg_mult,
            self.service.model_params.clip_out_dim,
        ]
        
        # These inputs will be reused for both front and back operations
        denoise_inputs = {
            "img": sfnp.device_array.for_device(
                device, img_shape, self.service.model_params.sampler_dtype
            ),
            "txt": sfnp.device_array.for_device(
                device, txt_shape, self.service.model_params.sampler_dtype
            ),
            "vec": sfnp.device_array.for_device(
                device, vec_shape, self.service.model_params.sampler_dtype
            ),
            "step": sfnp.device_array.for_device(device, [1], sfnp.int64),
            "timesteps": sfnp.device_array.for_device(
                device, [100], self.service.model_params.sampler_dtype
            ),
            "guidance_scale": sfnp.device_array.for_device(
                device, [req_bs], self.service.model_params.sampler_dtype
            ),
        }
        
        # We'll need to store intermediate state between front/back
        intermediate_state = sfnp.device_array.for_device(
            device, img_shape, self.service.model_params.sampler_dtype
        )
        
        # Setup all the input tensors just like in the regular denoise function
        gs_host = denoise_inputs["guidance_scale"].for_transfer()
        sample_host = sfnp.device_array.for_host(
            device, img_shape, self.service.model_params.sampler_dtype
        )
        guidance_float = sfnp.device_array.for_host(device, [req_bs], sfnp.float32)

        for i in range(req_bs):
            guidance_float.view(i).items = [requests[i].guidance_scale]
            cfg_dim = i * cfg_mult

            # Set up sample latent inputs
            req_samp = requests[i].sample
            for rep in range(cfg_mult):
                sample_host.view(slice(cfg_dim + rep, cfg_dim + rep + 1)).copy_from(
                    req_samp
                )
            denoise_inputs["img"].view(slice(cfg_dim, cfg_dim + cfg_mult)).copy_from(
                sample_host
            )

            # Set up t5xxl hidden states
            txt = requests[i].txt
            if (
                self.service.model_params.t5xxl_dtype
                != self.service.model_params.sampler_dtype
            ):
                inter = sfnp.device_array.for_host(
                    device, txt_shape, dtype=self.service.model_params.sampler_dtype
                )
                host = sfnp.device_array.for_host(
                    device, txt_shape, dtype=self.service.model_params.t5xxl_dtype
                )
                host.view(slice(cfg_dim, cfg_dim + cfg_mult)).copy_from(txt)
                await device
                sfnp.convert(
                    host,
                    dtype=self.service.model_params.sampler_dtype,
                    out=inter,
                )
                denoise_inputs["txt"].view(
                    slice(cfg_dim, cfg_dim + cfg_mult)
                ).copy_from(inter)
            else:
                denoise_inputs["txt"].view(
                    slice(cfg_dim, cfg_dim + cfg_mult)
                ).copy_from(txt)

            # Set up CLIP projections
            vec = requests[i].vec
            if (
                self.service.model_params.t5xxl_dtype
                != self.service.model_params.sampler_dtype
            ):
                for nc in range(cfg_mult):
                    inter = sfnp.device_array.for_host(
                        device, vec_shape, dtype=self.service.model_params.sampler_dtype
                    )
                    host = sfnp.device_array.for_host(
                        device, vec_shape, dtype=self.service.model_params.clip_dtype
                    )
                    host.view(slice(nc, nc + 1)).copy_from(vec)
                    await device
                    sfnp.convert(
                        host,
                        dtype=self.service.model_params.sampler_dtype,
                        out=inter,
                    )
                    denoise_inputs["vec"].view(slice(nc, nc + 1)).copy_from(inter)
            else:
                for nc in range(cfg_mult):
                    denoise_inputs["vec"].view(slice(nc, nc + 1)).copy_from(vec)
        
        # Convert guidance scale
        sfnp.convert(
            guidance_float, dtype=self.service.model_params.sampler_dtype, out=gs_host
        )
        denoise_inputs["guidance_scale"].copy_from(gs_host)
        
        # Set up timesteps
        await device
        ts_host = denoise_inputs["timesteps"].for_transfer()
        ts_float = sfnp.device_array.for_host(
            device, denoise_inputs["timesteps"].shape, dtype=sfnp.float32
        )
        with ts_float.map(write=True) as m:
            m.fill(float(1))
        for tstep in range(len(requests[0].timesteps)):
            with ts_float.view(tstep).map(write=True, discard=True) as m:
                m.fill(np.asarray(requests[0].timesteps[tstep], dtype="float32"))

        sfnp.convert(
            ts_float, dtype=self.service.model_params.sampler_dtype, out=ts_host
        )
        denoise_inputs["timesteps"].copy_from(ts_host)
        await device

        # Process each step, alternating between front and back halves
        for i, t in tqdm(
            enumerate(range(step_count)),
            disable=(not self.service.show_progress),
            desc=f"SPLIT DENOISE (bs{req_bs})",
        ):
            # Set step number
            s_host = denoise_inputs["step"].for_transfer()
            with s_host.map(write=True) as m:
                s_host.items = [i]
            denoise_inputs["step"].copy_from(s_host)
            
            # First, load and run the front half of the model
            logger.info(f"Step {i}: Running denoise_front")
            await self.sequential_service.load_component("sampler_front", self.worker_index)
            
            # Get the appropriate entrypoints for the front half
            entrypoints_front = self.sequential_service.inference_functions[self.worker_index]["denoise_front"]
            if req_bs not in list(entrypoints_front.keys()):
                raise ValueError(f"No function for batch size {req_bs} in denoise_front")
            
            fns_front = entrypoints_front[req_bs]
            
            # Run the front half with the current inputs
            await device
            logger.info("INVOKE front half sampler")
            (front_output,) = await fns_front["sampler_front"](
                *denoise_inputs.values(), fiber=self.fiber
            )
            await device
            
            # Store intermediate results in our buffer
            intermediate_state.copy_from(front_output)
            
            # Unload front half to free memory
            await self.sequential_service.unload_component("sampler_front", self.worker_index)
            
            # Now load and run the back half of the model
            logger.info(f"Step {i}: Running denoise_back")
            await self.sequential_service.load_component("sampler_back", self.worker_index)
            
            # Get the appropriate entrypoints for the back half
            entrypoints_back = self.sequential_service.inference_functions[self.worker_index]["denoise_back"]
            if req_bs not in list(entrypoints_back.keys()):
                raise ValueError(f"No function for batch size {req_bs} in denoise_back")
            
            fns_back = entrypoints_back[req_bs]
            
            # We need to feed the intermediate state as input to the back half
            denoise_inputs["img"].copy_from(intermediate_state)
            
            # Run the back half
            await device
            logger.info("INVOKE back half sampler")
            (back_output,) = await fns_back["sampler_back"](
                *denoise_inputs.values(), fiber=self.fiber
            )
            await device
            
            # Use the back output as input for the next step
            denoise_inputs["img"].copy_from(back_output)
            
            # Unload back half
            await self.sequential_service.unload_component("sampler_back", self.worker_index)

        # Set the denoised latents for each request (similar to _denoise)
        for idx, req in enumerate(requests):
            req.denoised_latents = sfnp.device_array.for_device(
                device, img_shape, self.service.model_params.vae_dtype
            )
            if (
                self.service.model_params.vae_dtype
                != self.service.model_params.sampler_dtype
            ):
                pred_shape = [
                    1,
                    (requests[0].height) * (requests[0].width) // 256,
                    64,
                ]
                denoised_inter = sfnp.device_array.for_host(
                    device, pred_shape, dtype=self.service.model_params.vae_dtype
                )
                denoised_host = sfnp.device_array.for_host(
                    device, pred_shape, dtype=self.service.model_params.sampler_dtype
                )
                denoised_host.copy_from(denoise_inputs["img"].view(idx * cfg_mult))
                await device
                sfnp.convert(
                    denoised_host,
                    dtype=self.service.model_params.vae_dtype,
                    out=denoised_inter,
                )
                req.denoised_latents.copy_from(denoised_inter)
            else:
                req.denoised_latents.copy_from(
                    denoise_inputs["img"].view(idx * cfg_mult)
                )
        return
