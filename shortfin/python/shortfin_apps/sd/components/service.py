# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import time
import numpy as np
import re
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from collections import namedtuple
import base64
import gc

import shortfin as sf
import shortfin.array as sfnp

from .config_struct import ModelParams
from .manager import SystemManager
from .messages import InferenceExecRequest, InferencePhase, StrobeMessage
from .tokenizer import Tokenizer
from .metrics import measure, log_duration_str

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
        use_batcher: bool = True,
    ):
        self.name = name

        # Application objects.
        self.sysman = sysman
        self.tokenizers = tokenizers
        self.model_params = model_params
        self.inference_parameters: dict[str, list[sf.BaseProgramParameters]] = {}
        self.inference_modules: dict[str, dict[int, sf.ProgramModule]] = {}
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
        self.meta_fibers = []
        self.idle_meta_fibers = []
        # For each worker index we create one on each device, and add their fibers to the idle set.
        # This roughly ensures that the first picked fibers are distributed across available devices.
        for idx, device in enumerate(self.sysman.ls.devices):
            for i in range(self.workers_per_device):
                worker = sysman.ls.create_worker(f"{name}-inference-{device.name}-{i}")
                self.workers.append(worker)
            for i in range(self.fibers_per_device):
                worker_idx = idx * workers_per_device + i % workers_per_device
                tgt_worker = self.workers[worker_idx]
                raw_fiber = sysman.ls.create_fiber(tgt_worker, devices=[device])
                meta_fiber = self.equip_fiber(
                    raw_fiber, len(self.meta_fibers), worker_idx
                )
                self.meta_fibers.append(meta_fiber)
                self.idle_meta_fibers.append(meta_fiber)
        for idx in range(len(self.workers)):
            self.inference_programs[idx] = {}
            self.inference_functions[idx] = {}

        # Scope dependent objects.
        self.use_batcher = use_batcher
        if self.use_batcher:
            self.batcher = BatcherProcess(self)

    def equip_fiber(self, fiber, idx, worker_idx):
        MetaFiber = namedtuple(
            "MetaFiber", ["fiber", "idx", "worker_idx", "device", "command_buffers"]
        )
        cb_sets_per_fiber = 1
        cbs = []
        for _ in range(cb_sets_per_fiber):
            for bs in self.model_params.all_batch_sizes:
                cbs.append(initialize_command_buffer(fiber, self.model_params, bs=bs))

        return MetaFiber(fiber, idx, worker_idx, fiber.device(0), cbs)

    def load_inference_module(
        self, vmfb_path: Path, component: str = None, batch_size: int = None
    ):
        if not self.inference_modules.get(component):
            self.inference_modules[component] = {}
        if batch_size:
            bs = batch_size
        else:
            match = re.search(r"_bs(\d+)_", str(vmfb_path))
            if match:
                bs = int(match.group(1))
            else:
                raise ValueError(
                    "Batch size not found in filename or provided to load function."
                )
        if not self.inference_modules[component].get(bs):
            self.inference_modules[component][bs] = []
        self.inference_modules[component][bs].append(
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
        for component in self.inference_modules.keys():
            for worker_idx, worker in enumerate(self.workers):
                self.inference_programs[worker_idx][component] = {}
            for batch_size in self.inference_modules[component]:
                component_modules = [
                    sf.ProgramModule.parameter_provider(
                        self.sysman.ls, *self.inference_parameters.get(component, [])
                    ),
                    *self.inference_modules[component][batch_size],
                ]

                for worker_idx, worker in enumerate(self.workers):
                    worker_devices = self.meta_fibers[
                        worker_idx * (self.fibers_per_worker)
                    ].fiber.raw_devices
                    logger.info(
                        f"Loading inference program: {component}, batch size {batch_size}, worker index: {worker_idx}, device: {worker_devices}"
                    )
                    self.inference_programs[worker_idx][component][
                        batch_size
                    ] = sf.Program(
                        modules=component_modules,
                        devices=worker_devices,
                        isolation=self.prog_isolation,
                        trace_execution=self.trace_execution,
                    )

        for worker_idx, worker in enumerate(self.workers):
            self.inference_functions[worker_idx]["encode"] = {}
            self.inference_functions[worker_idx]["denoise"] = {}
            self.inference_functions[worker_idx]["decode"] = {}

            for submodel, module_name in self.model_params.module_names.items():
                for bs in self.model_params.batch_sizes[submodel]:
                    if submodel == "clip":
                        self.inference_functions[worker_idx]["encode"][bs] = {}
                        fn_dest = self.inference_functions[worker_idx]["encode"][bs]
                    elif submodel in ["unet", "scheduled_unet", "scheduler"]:
                        if not self.inference_functions[worker_idx]["denoise"].get(bs):
                            self.inference_functions[worker_idx]["denoise"][bs] = {}
                        fn_dest = self.inference_functions[worker_idx]["denoise"][bs]
                    elif submodel == "vae":
                        self.inference_functions[worker_idx]["decode"][bs] = {}
                        fn_dest = self.inference_functions[worker_idx]["decode"][bs]
                    else:
                        raise ValueError("Received an unsupported submodel name.")
                    for fn in self.model_params.function_names[submodel]:
                        fn_dest[fn] = self.inference_programs[worker_idx][submodel][bs][
                            ".".join([module_name, fn])
                        ]
        if self.use_batcher:
            self.batcher.launch()

    def shutdown(self):
        if self.use_batcher:
            self.batcher.shutdown()
            del self.batcher
        del self.idle_meta_fibers
        del self.meta_fibers
        del self.workers
        gc.collect()

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
        super().__init__(fiber=service.meta_fibers[0].fiber)
        self.service = service
        self.batcher_infeed = self.system.create_queue()
        self.pending_requests: set[InferenceExecRequest] = set()
        self.strobe_enabled = True
        self.strobes: int = 0
        self.ideal_batch_size: int = max(service.model_params.batch_sizes["clip"])
        self.num_fibers = len(service.meta_fibers)

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

            await self.board_flights()

            self.strobe_enabled = True
        await strober_task

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
            if len(self.service.idle_meta_fibers) == 0:
                logger.debug("Waiting for an idle fiber...")
                return
            meta_fiber = self.service.idle_meta_fibers.pop(0)
            logger.debug(
                f"Sending batch to fiber {meta_fiber.idx} (worker {meta_fiber.worker_idx})"
            )
            await self.board(batch["reqs"][0], meta_fiber=meta_fiber)
            if self.service.prog_isolation != sf.ProgramIsolation.PER_FIBER:
                self.service.idle_meta_fibers.append(meta_fiber)

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

    async def board(self, request, meta_fiber):
        exec_process = InferenceExecutorProcess(self.service, meta_fiber)
        exec_process.exec_request = request
        self.pending_requests.remove(request)
        exec_process.launch()


########################################################################################
# Inference Executors
########################################################################################


class InferenceExecutorProcess(sf.Process):
    """Executes a stable diffusion inference batch"""

    def __init__(
        self,
        service: GenerateService,
        meta_fiber,
    ):
        super().__init__(fiber=meta_fiber.fiber)
        self.service = service
        self.meta_fiber = meta_fiber
        self.worker_index = meta_fiber.worker_idx
        self.exec_request: InferenceExecRequest = None

    def assign_command_buffer(self, request: InferenceExecRequest):
        for cb in self.meta_fiber.command_buffers:
            if cb.batch_size == self.exec_request.batch_size:
                self.exec_request.set_command_buffer(cb)
                self.meta_fiber.command_buffers.remove(cb)
                return
        cb = initialize_command_buffer(
            self.fiber, self.service.model_params, request.batch_size
        )
        self.exec_request.set_command_buffer(cb)
        return

    @measure(type="exec", task="inference process")
    async def run(self):
        try:
            if not self.exec_request.command_buffer:
                self.assign_command_buffer(self.exec_request)

            device = self.fiber.device(0)
            phases = self.exec_request.phases
            if phases[InferencePhase.PREPARE]["required"]:
                await self._prepare(device=device)
            if phases[InferencePhase.ENCODE]["required"]:
                await self._encode(device=device)
            if phases[InferencePhase.DENOISE]["required"]:
                await self._denoise(device=device)
            if phases[InferencePhase.DECODE]["required"]:
                await self._decode(device=device)
            else:
                # Decode and postprocess both need the output data to be on the host.
                # With decode enabled, decode itself will wait for the data.
                # With decode disabled, whether or not we're postprocessing,
                # we're done with the GPU, so we wait for it to finish here.
                await device
            if phases[InferencePhase.POSTPROCESS]["required"]:
                await self._postprocess(device=device)
            self.exec_request.done.set_success()

        except Exception:
            logger.exception("Fatal error in image generation")
            # TODO: Cancel and set error correctly
            self.exec_request.done.set_success()

        self.meta_fiber.command_buffers.append(self.exec_request.command_buffer)
        if self.service.prog_isolation == sf.ProgramIsolation.PER_FIBER:
            self.service.idle_meta_fibers.append(self.meta_fiber)

    async def _prepare(self, device):
        # Tokenize prompts and negative prompts. We tokenize in bs1 for now and join later.
        # Tokenize the prompts if the request does not hold input_ids.
        batch_ids_lists = []
        cb = self.exec_request.command_buffer
        if isinstance(self.exec_request.prompt, str):
            self.exec_request.prompt = [self.exec_request.prompt]
        if isinstance(self.exec_request.neg_prompt, str):
            self.exec_request.neg_prompt = [self.exec_request.neg_prompt]
        for i in range(self.exec_request.batch_size):
            input_ids_list = []
            neg_ids_list = []
            for tokenizer in self.service.tokenizers:
                input_ids = tokenizer.encode(self.exec_request.prompt[i]).input_ids
                input_ids_list.append(input_ids)
                neg_ids = tokenizer.encode(self.exec_request.neg_prompt[i]).input_ids
                neg_ids_list.append(neg_ids)
            ids_list = [*input_ids_list, *neg_ids_list]
            batch_ids_lists.append(ids_list)

        # Prepare tokenized input ids for CLIP inference
        host_arrs = [None] * len(cb.input_ids)
        for idx, arr in enumerate(cb.input_ids):
            host_arrs[idx] = arr.for_transfer()
            for i in range(self.exec_request.batch_size):
                with host_arrs[idx].view(i).map(write=True, discard=True) as m:

                    # TODO: fix this attr redundancy
                    np_arr = batch_ids_lists[i][idx]

                    m.fill(np_arr)
            cb.input_ids[idx].copy_from(host_arrs[idx])

        # Generate random sample latents.
        seed = self.exec_request.seed

        # Create and populate sample device array.
        generator = sfnp.RandomGenerator(seed)

        sample_host = cb.sample.for_transfer()
        with sample_host.map(discard=True) as m:
            m.fill(bytes(1))

        sfnp.fill_randn(sample_host, generator=generator)

        cb.sample.copy_from(sample_host)
        return

    async def _encode(self, device):
        req_bs = self.exec_request.batch_size
        entrypoints = self.service.inference_functions[self.worker_index]["encode"]
        assert req_bs in list(entrypoints.keys())
        for bs, fns in entrypoints.items():
            if bs == req_bs:
                break
        cb = self.exec_request.command_buffer
        # Encode tokenized inputs.
        logger.debug(
            "INVOKE %r: %s",
            fns["encode_prompts"],
            "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(cb.input_ids)]),
        )
        cb.prompt_embeds, cb.text_embeds = await fns["encode_prompts"](
            *cb.input_ids, fiber=self.fiber
        )
        return

    async def _denoise(self, device):
        req_bs = self.exec_request.batch_size
        entrypoints = self.service.inference_functions[self.worker_index]["denoise"]
        assert req_bs in list(entrypoints.keys())
        for bs, fns in entrypoints.items():
            if bs == req_bs:
                break

        cb = self.exec_request.command_buffer

        logger.debug(
            "INVOKE %r",
            fns["run_initialize"],
        )
        (cb.latents, cb.time_ids, cb.timesteps, cb.sigmas,) = await fns[
            "run_initialize"
        ](cb.sample, cb.num_steps, fiber=self.fiber)
        accum_step_duration = 0  # Accumulated duration for all steps

        for i, t in tqdm(
            enumerate(range(self.exec_request.steps)),
            disable=(not self.service.show_progress),
            desc=f"DENOISE (bs{req_bs})",
        ):
            start = time.time()
            step = cb.steps_arr.view(i)
            if self.service.model_params.use_scheduled_unet:
                logger.debug(
                    "INVOKE %r",
                    fns["run_forward"],
                )
                (cb.latents,) = await fns["run_forward"](
                    cb.latents,
                    cb.prompt_embeds,
                    cb.text_embeds,
                    cb.time_ids,
                    cb.guidance_scale,
                    step,
                    cb.timesteps,
                    cb.sigmas,
                    fiber=self.fiber,
                )
            else:
                logger.debug(
                    "INVOKE %r",
                    fns["run_scale"],
                )
                (cb.latent_model_input, cb.t, cb.sigma, cb.next_sigma,) = await fns[
                    "run_scale"
                ](cb.latents, step, cb.timesteps, cb.sigmas, fiber=self.fiber)
                logger.debug(
                    "INVOKE %r",
                    fns["main"],
                )
                (cb.noise_pred,) = await fns["main"](
                    cb.latent_model_input,
                    cb.t,
                    cb.prompt_embeds,
                    cb.text_embeds,
                    cb.time_ids,
                    cb.guidance_scale,
                    fiber=self.fiber,
                )
                logger.debug(
                    "INVOKE %r",
                    fns["run_step"],
                )
                (cb.latents,) = await fns["run_step"](
                    cb.noise_pred, cb.latents, cb.sigma, cb.next_sigma, fiber=self.fiber
                )
            duration = time.time() - start
            accum_step_duration += duration
        average_step_duration = accum_step_duration / self.exec_request.steps
        log_duration_str(
            average_step_duration, "denoise (UNet) single step average", req_bs
        )
        return

    async def _decode(self, device):
        req_bs = self.exec_request.batch_size
        prog_bs = req_bs
        cb = self.exec_request.command_buffer
        # Decode latents to images
        entrypoints = self.service.inference_functions[self.worker_index]["decode"]
        if req_bs not in list(entrypoints.keys()):
            prog_bs = 1
        for bs, fns in entrypoints.items():
            if bs == prog_bs:
                break
            raise RuntimeError(f"Decode program batch size {prog_bs} not found.")
        if req_bs != prog_bs:
            for i in range(req_bs):
                # Decode the denoised latents.
                logger.debug(
                    "INVOKE %r: %s",
                    fns["decode"],
                    "".join([f"\n  0: {cb.latents.shape}"]),
                )
                res = cb.images.view(i)
                (res,) = await fns["decode"](cb.latents.view(i), fiber=self.fiber)
                cb.images.view(i).copy_from(res)
        else:
            logger.debug(
                "INVOKE %r: %s",
                fns["decode"],
                "".join([f"\n  0: {cb.latents.shape}"]),
            )
            (cb.images,) = await fns["decode"](cb.latents, fiber=self.fiber)
        cb.images_host.copy_from(cb.images)

        # Wait for the device-to-host transfer, so that we can read the
        # data with .items.
        await device

        image_array = cb.images_host.items
        dtype = image_array.typecode
        if cb.images_host.dtype == sfnp.float16:
            dtype = np.float16
        self.exec_request.image_array = np.frombuffer(image_array, dtype=dtype).reshape(
            self.exec_request.batch_size,
            3,
            self.exec_request.height,
            self.exec_request.width,
        )
        return

    async def _postprocess(self, device):
        # Process output images
        # TODO: reimpl with sfnp
        permuted = np.transpose(self.exec_request.image_array, (0, 2, 3, 1))[0]
        cast_image = (permuted * 255).round().astype("uint8")
        image_bytes = Image.fromarray(cast_image).tobytes()

        image = base64.b64encode(image_bytes).decode("utf-8")
        self.exec_request.result_image = image
        return


def initialize_command_buffer(fiber, model_params: ModelParams, bs: int = 1):
    device = fiber.device(0)
    h = model_params.dims[0][0]
    w = model_params.dims[0][1]
    c = model_params.num_latents_channels
    cfg_bs = bs * 2
    datas = {
        # CLIP
        "input_ids": [
            sfnp.device_array.for_device(
                device, [bs, model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [bs, model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [bs, model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [bs, model_params.max_seq_len], sfnp.sint64
            ),
        ],
        # DENOISE
        "prompt_embeds": sfnp.device_array.for_device(
            device, [cfg_bs, model_params.max_seq_len, 2048], model_params.unet_dtype
        ),
        "text_embeds": sfnp.device_array.for_device(
            device, [cfg_bs, 1280], model_params.unet_dtype
        ),
        "sample": sfnp.device_array.for_device(
            device, [bs, c, h // 8, w // 8], model_params.unet_dtype
        ),
        "latents": sfnp.device_array.for_device(
            device, [bs, c, h // 8, w // 8], model_params.unet_dtype
        ),
        "noise_pred": sfnp.device_array.for_device(
            device, [bs, c, h // 8, w // 8], model_params.unet_dtype
        ),
        "num_steps": sfnp.device_array.for_device(device, [1], sfnp.sint64),
        "steps_arr": sfnp.device_array.for_device(device, [100], sfnp.sint64),
        "timesteps": sfnp.device_array.for_device(device, [100], sfnp.float32),
        "sigmas": sfnp.device_array.for_device(device, [100], sfnp.float32),
        "latent_model_input": sfnp.device_array.for_device(
            device, [bs, c, h // 8, w // 8], model_params.unet_dtype
        ),
        "t": sfnp.device_array.for_device(device, [1], model_params.unet_dtype),
        "sigma": sfnp.device_array.for_device(device, [1], model_params.unet_dtype),
        "next_sigma": sfnp.device_array.for_device(
            device, [1], model_params.unet_dtype
        ),
        "time_ids": sfnp.device_array.for_device(
            device, [bs, 6], model_params.unet_dtype
        ),
        "guidance_scale": sfnp.device_array.for_device(
            device, [1], model_params.unet_dtype
        ),
        # VAE
        "images": sfnp.device_array.for_device(
            device, [bs, 3, h, w], model_params.vae_dtype
        ),
        "images_host": sfnp.device_array.for_host(
            device, [bs, 3, h, w], model_params.vae_dtype
        ),
    }

    class ServiceCmdBuffer:
        def __init__(self, d):
            self.__dict__ = d
            self.batch_size = bs

    cb = ServiceCmdBuffer(datas)
    return cb
