# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import numpy as np
from typing import Tuple, Union

import shortfin.array as sfnp

from abc import ABC, abstractmethod
from asyncio import gather
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set
from uuid import uuid4

from .config import DecodeConfig, LogitsNormalization, TokenSelectionStrategyConfig
from .sampler import Sampler
from ..messages import LlmInferenceExecRequest

logger = logging.getLogger(__name__)

TOP_P_DEFAULT_SELECTION = 32


@dataclass
class BaseBeam(ABC):
    exec_req: LlmInferenceExecRequest

    decode_config: DecodeConfig

    sampler: Sampler = field(default_factory=Sampler)

    score: float = 0.0
    accumulated_normalization: float = 0.0
    last_token: int | None = None

    @abstractmethod
    def sample_default(
        self, logits: np.array, indices: Union[np.array, None], num_completed_beams: int
    ):
        """Sample the logits using the default sampling strategy for a give `Beam`.

        Args:
            logits (np.array): The logits from which to select.
            indices (np.array | None): Optional pre-selected indices.
        """

    @abstractmethod
    def sample_top_k(
        self,
        logits: np.array,
        indices: Union[np.array, None],
        top_k: int,
        num_completed_beams: int,
    ):
        """Sample the top-k tokens from the logits.

        Args:
            logits (np.array): The logits from which to select.
            indices (np.array | None): Optional pre-selected indices.
            top_k (int): The number of top tokens to select.
            num_completed_beams (int): Number of completed beams.
        """

    @abstractmethod
    def sample_top_p(
        self,
        tokens: np.array,
        probs: np.array,
        top_p: float,
        num_completed_beams: int,
    ) -> Tuple[np.array, np.array]:
        """Sample the top-p tokens from the logits.

        Args:
            tokens (np.array): The tokens to sample from.
            probs (np.array): The probabilities of the tokens.
            top_p (float): The cumulative probability threshold for sampling.
            num_completed_beams (int): Number of completed beams.
        """

    @abstractmethod
    def get_results(
        self, tokens: np.array, probs: np.array
    ) -> Tuple[np.array, np.array]:
        """Convert the results to a format suitable for the beam.

        Args:
            tokens (np.array): The selected tokens.
            probs (np.array): The probabilities of the selected tokens.
        """

    @staticmethod
    def replicate_inference_exec_requests(
        exec_req: LlmInferenceExecRequest,
        replicate: int,
    ) -> List[LlmInferenceExecRequest]:
        """Replicate inference requests for each beam."""
        exec_reqs = [exec_req]
        for _ in range(replicate):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))

        return exec_reqs

    @classmethod
    def clone(cls, beam: "BaseBeam") -> "BaseBeam":
        return cls(
            exec_req=beam.exec_req,
            score=beam.score,
            accumulated_normalization=beam.accumulated_normalization,
            last_token=beam.last_token,
            decode_config=beam.decode_config,
        )

    def sample_logits(self, num_completed_beams: int):
        """Define how to sample and select tokens for a give `Beam`"""
        exec_req = self.exec_req
        decode_config = self.decode_config

        top_k = decode_config.top_k
        top_p = decode_config.top_p

        decode_config = self.decode_config
        logits = np.array(exec_req.result_logits)
        indices = exec_req.result_indices

        if (top_k, top_p) == (None, None):
            return self.sample_default(logits, indices, num_completed_beams)

        indices = np.array(indices) if indices is not None else None
        if top_k is not None:
            tokens, probs = self.sample_top_k(
                logits,
                indices,
                top_k,
                num_completed_beams,
            )
        else:
            tokens, probs = logits, indices

        if top_p is not None:
            tokens, probs = self.sample_top_p(tokens, probs, top_p, num_completed_beams)

        return self.get_results(tokens, probs)

    def update_exec_req(self):
        """Update the `LlmInferenceExecRequest` with the selected token."""
        self.exec_req.input_token_ids.append(self.last_token)
        self.exec_req.start_position += 1

    def apply_temperature(self, logits: np.array) -> np.array:
        """Apply temperature to the logits of a decode invocation.

        Args:
            temperature (float): Value to use for `temperature`.
        """
        if self.decode_config.temperature == 1.0:
            return logits
        return np.divide(logits, self.decode_config.temperature)

    def _softmax(self, logits: Union[np.array, sfnp.device_array]) -> np.array:
        if isinstance(logits, sfnp.device_array):
            logits = np.array(logits)

        x_max = np.max(logits)
        e_x = np.exp(logits - x_max)
        return e_x / np.sum(e_x)

    def _log_softmax(self, logits: Union[np.array, sfnp.device_array]) -> np.array:
        if isinstance(logits, sfnp.device_array):
            logits = np.array(logits)

        c = logits.max()
        shifted_logits = logits - c
        sumexp = np.log(np.exp(shifted_logits).sum())
        return shifted_logits - sumexp

    def convert_logits_normalization(
        self,
        current: LogitsNormalization,
        target: LogitsNormalization,
        logits: np.array,
        **kwargs,
    ) -> np.array:
        logits_conversion_map = {
            LogitsNormalization.NONE: {
                LogitsNormalization.LOG_SOFTMAX: self._log_softmax,
                LogitsNormalization.SOFTMAX: self._softmax,
                LogitsNormalization.NONE: lambda logits: logits,
            },
            LogitsNormalization.SOFTMAX: {
                LogitsNormalization.LOG_SOFTMAX: np.log,
                LogitsNormalization.SOFTMAX: lambda logits: logits,
            },
            LogitsNormalization.LOG_SOFTMAX: {
                LogitsNormalization.SOFTMAX: np.exp,
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

    def _pre_select_top_p(
        self, logits: np.array, indices: Union[np.array, None]
    ) -> Tuple[np.array, np.array]:
        top_p_selection = min(logits.shape[-1], TOP_P_DEFAULT_SELECTION)
        tokens, values = self.sampler.select_top_k(logits, indices, -top_p_selection)
        probs = self._to_softmax(
            values,
            self.decode_config.logits_normalization,
        )

        if indices is None:
            sorted_order = np.argsort(probs)[::-1]
            tokens = tokens[sorted_order]
            probs = probs[sorted_order]

        return tokens, probs

    def _to_softmax(
        self,
        values: np.array,
        logits_normalization: LogitsNormalization,
    ):
        probs = self.convert_logits_normalization(
            logits_normalization,
            LogitsNormalization.SOFTMAX,
            values,
        )

        return probs

    def _sample_logits_top_k(
        self,
        logits: np.array,
        indices: Union[np.array, None],
        top_k: int,
        num_selections: int,
    ):
        tokens, values = self.sampler.select_top_k(logits, indices, -top_k)

        probs = self._to_softmax(
            values,
            self.decode_config.logits_normalization,
        )

        if indices is None:
            sorted_order = np.argsort(probs)[::-1]
            tokens = tokens[sorted_order]
            probs = probs[sorted_order]

        return self.sampler.sample_top_k(
            tokens=tokens,
            probs=probs,
            k=num_selections,
        )

    def _sample_logits_top_p(
        self, tokens, probs, top_p, num_selections, return_probs: bool = False
    ):
        config = self.decode_config
        if config.top_k is None:
            tokens, probs = self._pre_select_top_p(tokens, probs)

        return self.sampler.sample_top_p(
            tokens=tokens,
            probs=probs,
            p=top_p,
            k=num_selections,
            return_probs=return_probs,
        )


class BeamSearchBeam(BaseBeam):
    def _convert_results_to_log_probs(
        self,
        probs: np.array,
    ):
        log_probs = self.convert_logits_normalization(
            LogitsNormalization.SOFTMAX,
            LogitsNormalization.LOG_SOFTMAX,
            probs,
        )

        return log_probs.tolist()

    def sample_default(
        self, logits: np.array, indices: Union[np.array, None], num_completed_beams: int
    ) -> Tuple[np.array, np.array]:
        """Sample the logits using the default sampling strategy for a beam search.

        Args:
            logits (np.array): The logits from which to select.
            indices (np.array | None): Optional pre-selected indices.

        Returns:
            Tuple[np.array, np.array]: The selected tokens and their probabilities.
        """
        k = self.decode_config.num_beams - num_completed_beams

        if indices is not None:
            indices = np.array(indices)

        tokens, probs = self.sampler.select_top_k(logits, indices, -k)

        if self.decode_config.logits_normalization == LogitsNormalization.NONE:
            probs = self.apply_temperature(probs)

        log_probs = self.convert_logits_normalization(
            self.decode_config.logits_normalization,
            LogitsNormalization.LOG_SOFTMAX,
            probs,
        ).tolist()

        return tokens, log_probs

    def sample_top_k(
        self,
        logits: np.array,
        indices: Union[np.array, None],
        top_k: int,
        num_completed_beams: int,
    ) -> Tuple[np.array, np.array]:
        return self._sample_logits_top_k(
            logits,
            indices,
            top_k,
            num_selections=self.decode_config.num_beams - num_completed_beams,
        )

    def sample_top_p(
        self,
        tokens: np.array,
        probs: np.array,
        top_p: float,
        num_completed_beams: int,
    ) -> Tuple[np.array, np.array]:
        """Sample the top-p tokens from the logits.

        Args:
            tokens (np.array): The tokens to sample from.
            probs (np.array): The probabilities of the tokens.
            top_p (float): The cumulative probability threshold for sampling.
            num_completed_beams (int): Number of completed beams.

        Returns:
            Tuple[np.array, np.array]: The selected tokens and their probabilities.
        """
        return self._sample_logits_top_p(
            tokens,
            probs,
            top_p,
            num_selections=self.decode_config.num_beams - num_completed_beams,
            return_probs=True,
        )

    def get_results(
        self, tokens: np.array, probs: np.array
    ) -> Tuple[np.array, np.array]:
        """Convert the results to log probabilities for beam search.

        Args:
            tokens (np.array): The selected tokens.
            probs (np.array): The probabilities of the selected tokens.

        Returns:
            Tuple[np.array, np.array]: The tokens and their log probabilities.
        """
        log_probs = self._convert_results_to_log_probs(probs)
        return tokens, log_probs


class DefaultBeam(BaseBeam):
    def sample_default(self, logits, indices, _):
        if indices is not None:
            return indices.items[0]

        return self.sampler.select_greedy(logits)

    def sample_top_k(self, logits, indices, top_k: int, _: int):
        decode_config = self.decode_config

        num_selections = 1 if decode_config.top_p is None else top_k

        return self._sample_logits_top_k(
            logits,
            indices,
            top_k,
            num_selections=num_selections,
        )

    def sample_top_p(
        self,
        tokens: np.array,
        probs: np.array,
        top_p: float,
        _,
    ) -> Tuple[np.array, np.array]:
        """Sample the top-p tokens from the logits.

        Args:
            tokens (np.array): The tokens to sample from.
            probs (np.array): The probabilities of the tokens.
            top_p (float): The cumulative probability threshold for sampling.
            num_completed_beams (int): Number of completed beams.

        Returns:
            Tuple[np.array, np.array]: The selected tokens and their probabilities.
        """
        return self._sample_logits_top_p(
            tokens,
            probs,
            top_p,
            num_selections=1,
        )

    def get_results(self, tokens, _):
        return int(tokens[0])


def build_beam_group(
    exec_req: LlmInferenceExecRequest,
    config: TokenSelectionStrategyConfig,
    selection_callback: Callable[[List[BaseBeam], List[BaseBeam]], List[BaseBeam]],
) -> Callable[[LlmInferenceExecRequest], BaseBeam]:
    """Select the appropriate beam class based on the decode configuration."""
    decode_config = config.decode_config
    if not decode_config.use_beam_search and decode_config.num_beams > 1:
        exec_reqs = BaseBeam.replicate_inference_exec_requests(
            exec_req,
            decode_config.num_beams - 1,
        )
    else:
        exec_reqs = [exec_req]

    beam_cls = BeamSearchBeam if decode_config.use_beam_search else DefaultBeam
    beams = [beam_cls(exec_req, decode_config=decode_config) for exec_req in exec_reqs]
    return BeamGroup(
        config.eos_token_id,
        decode_config.num_beams,
        beams,
        selection_callback,
    )


class BeamGroup:
    def __init__(
        self,
        eos_token_id: int,
        num_beams: int,
        beams: List[BaseBeam],
        selection_callback: Callable[
            [List[BaseBeam], List[BaseBeam]],
            List[BaseBeam],
        ],
    ):
        self.beam_group_id = str(uuid4())
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.active_beams = beams
        self.selection_callback = selection_callback
        self.completed_beams: List[BaseBeam] = []

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
        active_beams: List[BaseBeam] = []
        active_reqs: Set[LlmInferenceExecRequest] = set()
        completed_beams: List[BaseBeam] = []
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
