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

logger = logging.getLogger("shortfin-flux.service")

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
        clip_tokenizers: list[Tokenizer],
        t5xxl_tokenizers: list[Tokenizer],
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
        self.clip_tokenizers = clip_tokenizers
        self.t5xxl_tokenizers = t5xxl_tokenizers
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
            self.inference_functions[idx] = {
                "clip": {},
                "t5xxl": {},
                "denoise": {},
                "decode": {},
            }
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
                    f"{self.model_params.t5xxl_module_name}.forward_bs4"
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
                breakpoint()
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
            latents_shape = [
                1,
                (requests[0].height) * (requests[0].width) // 256,
                64,
            ]
            # latents_shape = (
            #     1,
            #     channels,
            #     request.height // 8,
            #     request.width // 8,
            # )

            # Create and populate sample device array.
            generator = sfnp.RandomGenerator(seed)
            request.sample = sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.sampler_dtype
            )

            sample_host = request.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(bytes(1))

            sfnp.fill_randn(sample_host, generator=generator)

            request.sample.copy_from(sample_host)
            await device
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
                [req_bs, self.service.model_params.clip_max_seq_len, 2],
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
        (vec, _) = await fn(*clip_inputs, fiber=self.fiber)

        await device
        for i in range(req_bs):
            cfg_mult = 2
            requests[i].vec = vec.view(slice(i * cfg_mult, (i + 1) * cfg_mult))

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
                device, [4, self.service.model_params.max_seq_len], sfnp.sint64
            ),
        ]
        host_arrs = [None]
        for idx, arr in enumerate(t5xxl_inputs):
            host_arrs[idx] = arr.for_transfer()
            for i in range(req_bs):
                np_arr = requests[i].t5xxl_input_ids[idx].input_ids
                for rep in range(4):
                    with host_arrs[idx].view(rep).map(write=True, discard=True) as m:
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
            cfg_mult = 2
            requests[i].txt = txt.view(slice(i * cfg_mult, (i + 1) * cfg_mult))

        return

    async def _denoise(self, device, requests):
        req_bs = len(requests)
        step_count = requests[0].steps
        cfg_mult = 2 if not self.service.model_params.is_schnell else 1
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
            self.service.model_params.max_seq_len,
            4096,
        ]
        vec_shape = [
            req_bs * cfg_mult,
            768,
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
            "num_steps": sfnp.device_array.for_device(device, [1], sfnp.int64),
            "guidance_scale": sfnp.device_array.for_device(
                device, [req_bs], self.service.model_params.sampler_dtype
            ),
        }
        # Send guidance scale to device.
        gs_host = denoise_inputs["guidance_scale"].for_transfer()
        sample_host = sfnp.device_array.for_host(
            device, img_shape, self.service.model_params.sampler_dtype
        )
        for i in range(req_bs):
            cfg_dim = i * cfg_mult
            with gs_host.view(i).map(write=True, discard=True) as m:
                # TODO: do this without numpy
                np_arr = np.asarray(requests[i].guidance_scale, dtype="float32")

                m.fill(np_arr)

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
            denoise_inputs["txt"].view(slice(cfg_dim, cfg_dim + cfg_mult)).copy_from(
                txt
            )

            # Batch CLIP projections.
            vec = requests[i].vec
            denoise_inputs["vec"].view(slice(cfg_dim, cfg_dim + cfg_mult)).copy_from(
                vec
            )

        denoise_inputs["guidance_scale"].copy_from(gs_host)

        ns_host = denoise_inputs["num_steps"].for_transfer()
        with ns_host.map(write=True) as m:
            ns_host.items = [step_count]

        denoise_inputs["num_steps"].copy_from(ns_host)

        for i, t in tqdm(
            enumerate(range(step_count)),
            disable=(not self.service.show_progress),
            desc=f"DENOISE (bs{req_bs})",
        ):
            s_host = denoise_inputs["step"].for_transfer()
            with s_host.map(write=True) as m:
                s_host.items = [i]
            denoise_inputs["step"].copy_from(s_host)

            logger.debug(
                "INVOKE %r",
                fns["sampler"],
            )
            (noise_pred,) = await fns["sampler"](
                *denoise_inputs.values(), fiber=self.fiber
            )

            denoise_inputs["img"].copy_from(noise_pred)

        for idx, req in enumerate(requests):
            req.denoised_latents = sfnp.device_array.for_device(
                device, img_shape, self.service.model_params.vae_dtype
            )
            req.denoised_latents.copy_from(denoise_inputs["img"].view(idx * cfg_mult))
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
        await device
        latents_shape = [
            req_bs,
            (requests[0].height) * (requests[0].width) // 256,
            64,
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
                1,
                3,
                req.height,
                req.width,
            ]
            images_planar = sfnp.device_array.for_host(
                device, image_shape, self.service.model_params.vae_dtype
            )
            images_planar.copy_from(req.image_array)
            for j in range(3):
                data = [0.3 + j * 0.1 for _ in range(req.height * req.width)]
                images_planar.view(0, j).items = data
            permuted = sfnp.transpose(images_planar, (0, 2, 3, 1))
            breakpoint()
            cast_image = sfnp.multiply(127.5, (sfnp.add(permuted, 1.0)))
            image = sfnp.round(cast_image, dtype=sfnp.uint8)

            image_bytes = bytes(image.map(read=True))

            image = base64.b64encode(image_bytes).decode("utf-8")
            req.result_image = image
        return
