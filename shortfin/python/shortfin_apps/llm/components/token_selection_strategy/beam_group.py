# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from abc import ABC, abstractmethod
from asyncio import gather
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set
from uuid import uuid4

import shortfin as sf
import shortfin.array as sfnp

from .base_token_selection_strategy import DecodeConfig
from .config import LogitsNormalization
from .sampler import Sampler
from ..messages import LlmInferenceExecRequest

from shortfin_apps.utils import (
    convert_int_to_float,
    convert_float_to_int,
    convert_list_to_device_array,
)

logger = logging.getLogger(__name__)


# TODO: Define `top_p` function in base class when enabled in
# shortfin.
@dataclass
class Beam(ABC):
    exec_req: LlmInferenceExecRequest

    decode_config: DecodeConfig

    sampler: Sampler = field(default_factory=Sampler)

    score: float = 0.0
    accumulated_normalization: float = 0.0
    last_token: int | None = None

    def apply_temperature(self, logits: sfnp.device_array):
        """Apply temperature to the logits of a decode invocation.

        Args:
            temperature (float): Value to use for `temperature`.
        """
        if self.decode_config.temperature == 1.0:
            return
        return sfnp.divide(logits, self.decode_config.temperature)

    def convert_logits_normalization(
        self,
        current: LogitsNormalization,
        target: LogitsNormalization,
        logits: sfnp.device_array,
        **kwargs,
    ) -> sfnp.device_array:
        logits_conversion_map = {
            LogitsNormalization.NONE: {
                LogitsNormalization.LOG_SOFTMAX: sfnp.log_softmax,
                LogitsNormalization.SOFTMAX: sfnp.softmax,
                LogitsNormalization.NONE: lambda logits: logits,
            },
            LogitsNormalization.SOFTMAX: {
                LogitsNormalization.LOG_SOFTMAX: sfnp.log,
                LogitsNormalization.SOFTMAX: lambda logits: logits,
            },
            LogitsNormalization.LOG_SOFTMAX: {
                LogitsNormalization.SOFTMAX: sfnp.exp,
                LogitsNormalization.LOG_SOFTMAX: lambda logits: logits,
            },
        }

        target_conversions = logits_conversion_map.get(current)
        if target_conversions is None:
            raise KeyError(f"Cannot convert current normalization: {current}")

        conversion_function = target_conversions.get(target)
        if conversion_function is None:
            raise KeyError(f"Cannot convert {current} to {target}")

        if kwargs:
            converted_logits = conversion_function(logits, **kwargs)
        else:
            converted_logits = conversion_function(logits)

        return converted_logits

    @abstractmethod
    def sample_logits(self):
        """Define how to sample and select tokens for a give `Beam`"""
        pass

    def _to_softmax(
        self,
        values: List,
        dtype: sfnp.DType,
        device: sf.ScopedDevice,
        logits_normalization: LogitsNormalization,
    ):
        if dtype in [sfnp.float16]:
            values = [convert_float_to_int(value, dtype) for value in values]

        probs_sf = convert_list_to_device_array(
            values,
            [len(values)],
            device,
            dtype,
        )

        if logits_normalization == LogitsNormalization.NONE:
            probs_sf = self.apply_temperature(probs_sf)

        probs = self.convert_logits_normalization(
            logits_normalization,
            LogitsNormalization.SOFTMAX,
            probs_sf,
            **{"device_visible": True},
        ).items.tolist()

        if dtype in [sfnp.float16]:
            probs = [convert_int_to_float(prob, dtype) for prob in probs]

        return probs

    def _sample_logits_top_k(self, logits: sfnp.device_array, top_k, num_selections):
        tokens, values = self.sampler.select_top_k(logits, -top_k)

        probs = self._to_softmax(
            values,
            logits.dtype,
            logits.device,
            self.decode_config.logits_normalization,
        )

        return self.sampler.sample_top_k(
            tokens,
            probs,
            k=num_selections,
        )

    def _sample_logits_top_p(self, tokens, probs, top_p, num_selections):
        return self.sampler.sample_top_p(
            tokens=tokens,
            probs=probs,
            p=top_p,
            k=num_selections,
        )

    @abstractmethod
    def update_score(self, value: float):
        """Update the score of a `beam`.

        Args:
            value (float): Value to update the score with.
        """
        pass

    @abstractmethod
    def update_exec_req(self):
        """Update an `LlmInferenceExecRequest`, after a decode loop"""
        pass

    @abstractmethod
    def normalize_score(self, value: float):
        """Normalize the score of a `beam`.

        Args:
            value (float): Value to normalize the score with.
        """
        pass

    @abstractmethod
    def update_final_score(self):
        """Define a `final_score` for a given beam, if applicable."""
        pass


class BeamGroup:
    def __init__(
        self,
        eos_token_id: int,
        num_beams: int,
        beams: List[Beam],
        selection_callback: Callable[
            [List[Beam], List[Beam]],
            List[Beam],
        ],
    ):
        self.beam_group_id = str(uuid4())
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.active_beams = beams
        self.selection_callback = selection_callback
        self.completed_beams: List[Beam] = []

    @property
    def active_beam_count(self):
        return len(self.active_beams)

    async def wait(self):
        done_signals = [beam.exec_req.done for beam in self.active_beams]
        return await gather(*done_signals)

    def process_beams(self):
        beam_selections = self.selection_callback(
            self.active_beams, self.completed_beams
        )
        visited_reqs: Dict[str, LlmInferenceExecRequest] = {}
        active_beams: List[Beam] = []
        active_reqs: Set[LlmInferenceExecRequest] = set()
        completed_beams: List[Beam] = []
        completed_reqs: Set[LlmInferenceExecRequest] = set()

        for i in range(len(beam_selections)):
            beam = beam_selections[i]
            new_req, token = beam.exec_req, beam.last_token

            if new_req.instance_id in visited_reqs:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = LlmInferenceExecRequest.copy_exec_request(visited_req)
                beam.exec_req = new_req

            visited_reqs[new_req.instance_id] = new_req
            if token == self.eos_token_id:
                completed_beams.append(beam)
                completed_reqs.add(new_req)
            else:
                active_beams.append(beam)
                active_reqs.add(new_req)

        for beam in completed_beams + active_beams:
            beam.update_exec_req()
            if beam.exec_req in completed_reqs:
                beam.exec_req.free_cache_pages()

        # Free cache pages of reqs we don't need anymore
        for beam in self.active_beams:
            if beam.exec_req not in active_reqs and beam.exec_req not in completed_reqs:
                beam.exec_req.free_cache_pages()

        self.active_beams = active_beams
        self.completed_beams.extend(completed_beams)

    def clean_up(self):
        logger.debug(f"Cleaning up {self.beam_group_id}...")

        # Ensure all requests have freed their cache pages
        for beam in self.active_beams + self.completed_beams:
            beam.exec_req.free_cache_pages()
