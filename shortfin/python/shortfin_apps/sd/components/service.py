# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import base64

import shortfin as sf
import shortfin.array as sfnp

from .config_struct import ModelParams
from .manager import SystemManager
from .messages import InferenceExecRequest, InferencePhase, StrobeMessage
from .tokenizer import Tokenizer
from .metrics import measure

logger = logging.getLogger("shortfin-sd.service")

prog_isolations = {
    "none": sf.ProgramIsolation.NONE,
    "per_fiber": sf.ProgramIsolation.PER_FIBER,
    "per_call": sf.ProgramIsolation.PER_CALL,
}


class GenerateService:
    """Top level service interface for image generation."""

    inference_programs: dict[str, sf.Program]

    inference_functions: dict[str, dict[str, sf.ProgramFunction]]

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        tokenizers: list[Tokenizer],
        model_params: ModelParams,
        fibers_per_device: int,
        workers_per_device: int = 1,
        prog_isolation: str = "per_fiber",
        show_progress: bool = False,
        trace_execution: bool = False,
    ):
        self.name = name

        # Application objects.
        self.sysman = sysman
        self.tokenizers = tokenizers
        self.model_params = model_params
        self.inference_parameters: dict[str, list[sf.BaseProgramParameters]] = {}
        self.inference_modules: dict[str, sf.ProgramModule] = {}
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
            self.inference_functions[idx] = {}
        # Scope dependent objects.
        self.batcher = BatcherProcess(self)

    def get_worker_index(self, fiber):
        if fiber not in self.fibers:
            raise ValueError("A worker was requested from a rogue fiber.")
        fiber_idx = self.fibers.index(fiber)
        worker_idx = int(
            (fiber_idx - fiber_idx % self.fibers_per_worker) / self.fibers_per_worker
        )
        return worker_idx

    def load_inference_module(self, vmfb_path: Path, component: str = None):
        if not self.inference_modules.get(component):
            self.inference_modules[component] = []
        self.inference_modules[component].append(
            sf.ProgramModule.load(self.sysman.ls, vmfb_path)
        )

    def load_inference_parameters(
        self,
        *paths: Path,
        parameter_scope: str,
        format: str = "",
        component: str = None,
    ):
        p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
        for path in paths:
            logger.info("Loading parameter fiber '%s' from: %s", parameter_scope, path)
            p.load(path, format=format)
        if not self.inference_parameters.get(component):
            self.inference_parameters[component] = []
        self.inference_parameters[component].append(p)

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
            self.inference_functions[worker_idx]["encode"] = {}
            for bs in self.model_params.clip_batch_sizes:
                self.inference_functions[worker_idx]["encode"][
                    bs
                ] = self.inference_programs[worker_idx]["clip"][
                    f"{self.model_params.clip_module_name}.encode_prompts"
                ]
            self.inference_functions[worker_idx]["denoise"] = {}
            for bs in self.model_params.unet_batch_sizes:
                self.inference_functions[worker_idx]["denoise"][bs] = {
                    "unet": self.inference_programs[worker_idx]["unet"][
                        f"{self.model_params.unet_module_name}.{self.model_params.unet_fn_name}"
                    ],
                    "init": self.inference_programs[worker_idx]["scheduler"][
                        f"{self.model_params.scheduler_module_name}.run_initialize"
                    ],
                    "scale": self.inference_programs[worker_idx]["scheduler"][
                        f"{self.model_params.scheduler_module_name}.run_scale"
                    ],
                    "step": self.inference_programs[worker_idx]["scheduler"][
                        f"{self.model_params.scheduler_module_name}.run_step"
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
# Batcher
########################################################################################


class BatcherProcess(sf.Process):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches.
    """

    STROBE_SHORT_DELAY = 0.5
    STROBE_LONG_DELAY = 1

    def __init__(self, service: GenerateService):
        super().__init__(fiber=service.fibers[0])
        self.service = service
        self.batcher_infeed = self.system.create_queue()
        self.pending_requests: set[InferenceExecRequest] = set()
        self.strobe_enabled = True
        self.strobes: int = 0
        self.ideal_batch_size: int = max(service.model_params.max_batch_size)
        self.num_fibers = len(service.fibers)

    def shutdown(self):
        self.batcher_infeed.close()

    def submit(self, request: StrobeMessage | InferenceExecRequest):
        self.batcher_infeed.write_nodelay(request)

    async def _background_strober(self):
        while not self.batcher_infeed.closed:
            await asyncio.sleep(
                BatcherProcess.STROBE_SHORT_DELAY
                if len(self.pending_requests) > 0
                else BatcherProcess.STROBE_LONG_DELAY
            )
            if self.strobe_enabled:
                self.submit(StrobeMessage())

    async def run(self):
        strober_task = asyncio.create_task(self._background_strober())
        reader = self.batcher_infeed.reader()
        while item := await reader():
            self.strobe_enabled = False
            if isinstance(item, InferenceExecRequest):
                self.pending_requests.add(item)
            elif isinstance(item, StrobeMessage):
                self.strobes += 1
            else:
                logger.error("Illegal message received by batcher: %r", item)

            self.board_flights()

            self.strobe_enabled = True
        await strober_task

    def board_flights(self):
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

    def sort_batches(self):
        """Files pending requests into sorted batches suitable for program invocations."""
        reqs = self.pending_requests
        next_key = 0
        batches = {}
        for req in reqs:
            is_sorted = False
            req_metas = [req.phases[phase]["metadata"] for phase in req.phases.keys()]

            for idx_key, data in batches.items():
                if not isinstance(data, dict):
                    logger.error(
                        "Expected to find a dictionary containing a list of requests and their shared metadatas."
                    )
                if len(batches[idx_key]["reqs"]) >= self.ideal_batch_size:
                    # Batch is full
                    next_key = idx_key + 1
                    continue
                elif data["meta"] == req_metas:
                    batches[idx_key]["reqs"].extend([req])
                    is_sorted = True
                    break
                else:
                    next_key = idx_key + 1
            if not is_sorted:
                batches[next_key] = {
                    "reqs": [req],
                    "meta": req_metas,
                }
        return batches

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
        self.exec_requests: list[InferenceExecRequest] = []

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
                await self._encode(device=device0, requests=self.exec_requests)
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
            input_ids_list = []
            neg_ids_list = []
            for tokenizer in self.service.tokenizers:
                input_ids = tokenizer.encode(request.prompt)
                input_ids_list.append(input_ids)
                neg_ids = tokenizer.encode(request.neg_prompt)
                neg_ids_list.append(neg_ids)
            ids_list = [*input_ids_list, *neg_ids_list]

            request.input_ids = ids_list

            # Generate random sample latents.
            seed = request.seed
            channels = self.service.model_params.num_latents_channels
            unet_dtype = self.service.model_params.unet_dtype
            latents_shape = (
                1,
                channels,
                request.height // 8,
                request.width // 8,
            )

            # Create and populate sample device array.
            generator = sfnp.RandomGenerator(seed)
            request.sample = sfnp.device_array.for_device(
                device, latents_shape, unet_dtype
            )

            sample_host = request.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(bytes(1))

            sfnp.fill_randn(sample_host, generator=generator)

            request.sample.copy_from(sample_host)
        return

    async def _encode(self, device, requests):
        req_bs = len(requests)
        entrypoints = self.service.inference_functions[self.worker_index]["encode"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._encode(device, [request])
            return
        for bs, fn in entrypoints.items():
            if bs == req_bs:
                break

        # Prepare tokenized input ids for CLIP inference

        clip_inputs = [
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
        ]
        host_arrs = [None] * 4
        for idx, arr in enumerate(clip_inputs):
            host_arrs[idx] = arr.for_transfer()
            for i in range(req_bs):
                with host_arrs[idx].view(i).map(write=True, discard=True) as m:

                    # TODO: fix this attr redundancy
                    np_arr = requests[i].input_ids[idx].input_ids

                    m.fill(np_arr)
            clip_inputs[idx].copy_from(host_arrs[idx])

        # Encode tokenized inputs.
        logger.debug(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(clip_inputs)]),
        )
        await device
        pe, te = await fn(*clip_inputs, fiber=self.fiber)

        for i in range(req_bs):
            cfg_mult = 2
            requests[i].prompt_embeds = pe.view(slice(i * cfg_mult, (i + 1) * cfg_mult))
            requests[i].text_embeds = te.view(slice(i * cfg_mult, (i + 1) * cfg_mult))

        return

    async def _denoise(self, device, requests):
        req_bs = len(requests)
        step_count = requests[0].steps
        cfg_mult = 2 if self.service.model_params.cfg_mode else 1
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
        latents_shape = [
            req_bs,
            self.service.model_params.num_latents_channels,
            requests[0].height // 8,
            requests[0].width // 8,
        ]
        # Assume we are doing classifier-free guidance
        hidden_states_shape = [
            req_bs * cfg_mult,
            self.service.model_params.max_seq_len,
            2048,
        ]
        text_embeds_shape = [
            req_bs * cfg_mult,
            1280,
        ]
        denoise_inputs = {
            "sample": sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.unet_dtype
            ),
            "encoder_hidden_states": sfnp.device_array.for_device(
                device, hidden_states_shape, self.service.model_params.unet_dtype
            ),
            "text_embeds": sfnp.device_array.for_device(
                device, text_embeds_shape, self.service.model_params.unet_dtype
            ),
            "guidance_scale": sfnp.device_array.for_device(
                device, [req_bs], self.service.model_params.unet_dtype
            ),
        }

        # Send guidance scale to device.
        gs_host = denoise_inputs["guidance_scale"].for_transfer()
        for i in range(req_bs):
            cfg_dim = i * cfg_mult
            with gs_host.view(i).map(write=True, discard=True) as m:
                # TODO: do this without numpy
                np_arr = np.asarray(requests[i].guidance_scale, dtype="float16")

                m.fill(np_arr)
            # Batch sample latent inputs on device.
            req_samp = requests[i].sample
            denoise_inputs["sample"].view(i).copy_from(req_samp)

            # Batch CLIP hidden states.
            enc = requests[i].prompt_embeds
            denoise_inputs["encoder_hidden_states"].view(
                slice(cfg_dim, cfg_dim + cfg_mult)
            ).copy_from(enc)

            # Batch CLIP text embeds.
            temb = requests[i].text_embeds
            denoise_inputs["text_embeds"].view(
                slice(cfg_dim, cfg_dim + cfg_mult)
            ).copy_from(temb)

        denoise_inputs["guidance_scale"].copy_from(gs_host)

        num_steps = sfnp.device_array.for_device(device, [1], sfnp.sint64)
        ns_host = num_steps.for_transfer()
        with ns_host.map(write=True) as m:
            ns_host.items = [step_count]
        num_steps.copy_from(ns_host)

        init_inputs = [
            denoise_inputs["sample"],
            num_steps,
        ]

        # Initialize scheduler.
        logger.debug(
            "INVOKE %r",
            fns["init"],
        )
        (latents, time_ids, timesteps, sigmas) = await fns["init"](
            *init_inputs, fiber=self.fiber
        )
        for i, t in tqdm(
            enumerate(range(step_count)),
            disable=(not self.service.show_progress),
            desc=f"DENOISE (bs{req_bs})",
        ):
            step = sfnp.device_array.for_device(device, [1], sfnp.sint64)
            s_host = step.for_transfer()
            with s_host.map(write=True) as m:
                s_host.items = [i]
            step.copy_from(s_host)
            scale_inputs = [latents, step, timesteps, sigmas]
            logger.debug(
                "INVOKE %r",
                fns["scale"],
            )
            latent_model_input, t, sigma, next_sigma = await fns["scale"](
                *scale_inputs, fiber=self.fiber
            )
            await device

            unet_inputs = [
                latent_model_input,
                t,
                denoise_inputs["encoder_hidden_states"],
                denoise_inputs["text_embeds"],
                time_ids,
                denoise_inputs["guidance_scale"],
            ]
            logger.debug(
                "INVOKE %r",
                fns["unet"],
            )
            (noise_pred,) = await fns["unet"](*unet_inputs, fiber=self.fiber)

            step_inputs = [noise_pred, latents, sigma, next_sigma]
            logger.debug(
                "INVOKE %r",
                fns["step"],
            )
            (latent_model_output,) = await fns["step"](*step_inputs, fiber=self.fiber)
            latents.copy_from(latent_model_output)

        for idx, req in enumerate(requests):
            req.denoised_latents = sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.vae_dtype
            )
            req.denoised_latents.copy_from(latents.view(idx))
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
            self.service.model_params.num_latents_channels,
            requests[0].height // 8,
            requests[0].width // 8,
        ]
        latents = sfnp.device_array.for_device(
            device, latents_shape, self.service.model_params.vae_dtype
        )
        for i in range(req_bs):
            latents.view(i).copy_from(requests[i].denoised_latents)

        await device
        # Decode the denoised latents.
        logger.debug(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  0: {latents.shape}"]),
        )
        (image,) = await fn(latents, fiber=self.fiber)

        await device
        images_shape = [
            req_bs,
            3,
            requests[0].height,
            requests[0].width,
        ]
        image_shape = [
            req_bs,
            3,
            requests[0].height,
            requests[0].width,
        ]
        images_host = sfnp.device_array.for_host(device, images_shape, sfnp.float16)
        images_host.copy_from(image)
        await device
        for idx, req in enumerate(requests):
            image_array = images_host.view(idx).items
            dtype = image_array.typecode
            if images_host.dtype == sfnp.float16:
                dtype = np.float16
            req.image_array = np.frombuffer(image_array, dtype=dtype).reshape(
                *image_shape
            )
        return

    async def _postprocess(self, device, requests):
        # Process output images
        for req in requests:
            # TODO: reimpl with sfnp
            permuted = np.transpose(req.image_array, (0, 2, 3, 1))[0]
            cast_image = (permuted * 255).round().astype("uint8")
            image_bytes = Image.fromarray(cast_image).tobytes()

            image = base64.b64encode(image_bytes).decode("utf-8")
            req.result_image = image
        return
