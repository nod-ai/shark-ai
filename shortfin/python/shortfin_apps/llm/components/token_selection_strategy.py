# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass, field, fields
from typing import Callable, List, Union, Tuple, Dict, Set, Optional, cast
from enum import Enum, auto
from abc import ABC, abstractmethod
from uuid import uuid4

from dataclasses_json import dataclass_json, Undefined

# --- External dependencies (assumed to be available in parent package) ---
from .io_struct import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_TEMPERATURE, NOT_PROVIDED
from .messages import LlmInferenceExecRequest, InferencePhase
import shortfin.array as sfnp

logger = logging.getLogger(__name__)

# --- Config and Enums ---
class LogitsNormalization(Enum):
    NONE = auto()
    SOFTMAX = auto()
    LOG_SOFTMAX = auto()

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            for member in cls:
                if member.name == value:
                    return member
        raise KeyError(f"Unknown token_selection_strategy: {value}")


def get_normalization_from_str(token_selection_strategy: str) -> LogitsNormalization:
    name_to_strategy = {
        strategy.name.lower(): strategy for strategy in LogitsNormalization
    }
    strategy = token_selection_strategy.lower()
    if strategy not in name_to_strategy:
        raise KeyError(f"Unknown token_selection_strategy: {token_selection_strategy}")
    return name_to_strategy[strategy]


class TokenSelectionStrategy(Enum):
    INDEPENDENT = auto()
    BEAM_SEARCH = auto()


def get_strategy_from_str(token_selection_strategy: str) -> TokenSelectionStrategy:
    name_to_strategy = {
        strategy.name.lower(): strategy for strategy in TokenSelectionStrategy
    }
    strategy = token_selection_strategy.lower()
    if strategy not in name_to_strategy:
        raise KeyError(f"Unknown token_selection_strategy: {token_selection_strategy}")
    return name_to_strategy[strategy]


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class DecodeConfig:
    num_beams: int = 1
    logits_normalization: LogitsNormalization = LogitsNormalization.NONE
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    use_beam_search: bool = False
    top_k: Optional[int] = None
    top_p: Optional[int] = None

    def update_from_sampling_params(self, sampling_params):
        for field in fields(sampling_params):
            if getattr(sampling_params, field.name) == NOT_PROVIDED:
                continue
            if hasattr(self, field.name):
                setattr(self, field.name, getattr(sampling_params, field.name))


@dataclass
class TokenSelectionStrategyConfig:
    decode_config: DecodeConfig
    prefill_callback: Callable[[LlmInferenceExecRequest], None]
    decode_callback: Callable[[LlmInferenceExecRequest], None]
    decode_begin_callback: Callable[[int], None]
    decode_end_callback: Callable[[int], None]
    results_callback: Callable[[Union[int, List[int]]], None]
    eos_token_id: int


# --- Sampler ---
@dataclass
class Sampler:
    def sample_top_k(
        self, tokens: np.ndarray, probs: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if probs is None:
            probs = np.array([])
        p = (
            probs / probs.sum()
            if len(probs) > 0
            else np.ones_like(tokens) / len(tokens)
        )
        choices = np.random.choice(tokens, size=k, replace=True, p=p)
        token_to_p = {int(t): float(p_) for t, p_ in zip(tokens, p)}
        chosen_probs = np.array([token_to_p[int(t)] for t in choices])
        return choices, chosen_probs

    def sample_top_p(
        self,
        tokens: np.ndarray,
        probs: np.ndarray,
        p: float,
        k: int,
        return_probs: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if probs is None or not hasattr(probs, "__len__") or len(probs) == 0:
            return np.array([]), np.array([])
        cum = np.cumsum(probs)
        idx = np.searchsorted(cum, p, side="right") + 1
        tokens, probs = tokens[:idx], probs[:idx]
        weights = (
            probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        )
        choices = np.random.choice(tokens, size=k, p=weights)
        if return_probs:
            prob_map = {tok: pr for tok, pr in zip(tokens, probs)}
            chosen_probs = np.array([prob_map[t] for t in choices])
        else:
            chosen_probs = np.zeros_like(choices, dtype=probs.dtype)
        return choices, chosen_probs

    def select_top_k(
        self,
        logits: Union[np.ndarray, object],
        indices: Optional[Union[np.ndarray, object]],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        if indices is not None and not isinstance(indices, np.ndarray):
            indices = np.array(indices)
        zero_indices = (0,) * (logits.ndim - 1)
        if indices is not None:
            tokens_index = zero_indices + (slice(None, -k),)
            return indices[tokens_index], logits[tokens_index]
        tokens_index = zero_indices + (slice(k, None),)
        partitioned_tokens = np.argpartition(logits, k)
        top_tokens = partitioned_tokens[tokens_index]
        top_values = np.take(logits, top_tokens, axis=-1)[zero_indices]
        return top_tokens, top_values

    def select_greedy(self, logits: np.ndarray) -> int:
        token = np.argmax(logits).item()
        return token


# --- Base Token Selection Strategy ---
@dataclass
class BaseTokenSelectionStrategy(ABC):
    token_selection_strategy_config: TokenSelectionStrategyConfig
    scorer: Optional[BaseBeamScorer]

    def _log_sampling_method(self):
        decode_config = self.token_selection_strategy_config.decode_config
        num_beams = decode_config.num_beams
        strategy = "indepdent" if not decode_config.use_beam_search else "beam_search"
        logger.debug(f"Using {strategy} selection method with {num_beams} beams...")
        if decode_config.top_k is not None:
            logger.debug(
                f"Using `top_k` sampling with `top_k == {decode_config.top_k}`"
            )
        if decode_config.top_p is not None:
            logger.debug(
                f"Using `top_p` sampling with `top_p == {decode_config.top_p}`"
            )

    def replicate_inference_exec_requests(
        self, exec_req: LlmInferenceExecRequest, replicate: int
    ) -> List[LlmInferenceExecRequest]:
        exec_reqs = [exec_req]
        for _ in range(replicate):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))
        return exec_reqs

    async def prefill(self, exec_req: LlmInferenceExecRequest) -> Optional[int]:
        if exec_req.status_tracker is None or exec_req.status_tracker.is_disconnected():
            return None
        token_selection_strategy_config = self.token_selection_strategy_config
        token_selection_strategy_config.prefill_callback(exec_req)
        await exec_req.done
        assert_message = f"{exec_req.instance_id}'s result_logits are None. This typically indicates an error during prefill invocation."
        assert exec_req.result_logits is not None, assert_message
        if exec_req.result_indices is not None:
            token_int = exec_req.result_indices.items[0]
        else:
            token = sfnp.argmax(exec_req.result_logits)
            token_int = token.items[0]
        decode_config = token_selection_strategy_config.decode_config
        if not decode_config.use_beam_search and decode_config.num_beams == 1:
            token_selection_strategy_config.results_callback(token_int)
        exec_req.input_token_ids.append(token_int)
        exec_req.start_position = len(exec_req.input_token_ids) - 1
        return token_int

    @abstractmethod
    async def decode(self, exec_req: LlmInferenceExecRequest) -> List[int]:
        pass


# --- Beam Group and Beams ---
TOP_P_DEFAULT_SELECTION = 32


@dataclass
class BaseBeam(ABC):
    exec_req: "LlmInferenceExecRequest"
    decode_config: "DecodeConfig"
    sampler: Sampler = field(default_factory=Sampler)
    score: float = 0.0
    accumulated_normalization: float = 0.0
    last_token: Optional[int] = None

    @abstractmethod
    def sample_default(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        num_completed_beams: int,
    ):
        pass

    @abstractmethod
    def sample_top_k(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        top_k: int,
        num_completed_beams: int,
    ):
        pass

    @abstractmethod
    def sample_top_p(
        self,
        tokens: np.ndarray,
        probs: np.ndarray,
        top_p: float,
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_results(
        self, tokens: np.ndarray, probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def replicate_inference_exec_requests(
        exec_req: LlmInferenceExecRequest, replicate: int
    ) -> List[LlmInferenceExecRequest]:
        exec_reqs = [exec_req]
        for _ in range(replicate):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))
        return exec_reqs

    @classmethod
    def clone(cls, beam: BaseBeam) -> BaseBeam:
        return cls(
            exec_req=beam.exec_req,
            score=beam.score,
            accumulated_normalization=beam.accumulated_normalization,
            last_token=beam.last_token,
            decode_config=beam.decode_config,
        )

    def sample_logits(self, num_completed_beams: int):
        exec_req = self.exec_req
        decode_config = self.decode_config
        top_k = decode_config.top_k
        top_p = decode_config.top_p
        logits = np.array(exec_req.result_logits)
        indices = exec_req.result_indices
        if (top_k, top_p) == (None, None):
            return self.sample_default(logits, indices, num_completed_beams)
        indices = np.array(indices) if indices is not None else None
        if top_k is not None:
            tokens, probs = self.sample_top_k(
                logits, indices, top_k, num_completed_beams
            )
        else:
            tokens, probs = logits, indices
        if top_p is not None:
            tokens, probs = self.sample_top_p(tokens, probs, top_p, num_completed_beams)
        return self.get_results(tokens, probs)

    def update_exec_req(self):
        self.exec_req.input_token_ids.append(self.last_token)
        self.exec_req.start_position += 1

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        if self.decode_config.temperature == 1.0:
            return logits
        return np.divide(logits, self.decode_config.temperature)

    def _softmax(self, logits: Union[np.ndarray, object]) -> np.ndarray:
        if not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        x_max = np.max(logits)
        e_x = np.exp(logits - x_max)
        return e_x / np.sum(e_x)

    def _log_softmax(self, logits: Union[np.ndarray, object]) -> np.ndarray:
        if not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        c = logits.max()
        shifted_logits = logits - c
        sumexp = np.log(np.exp(shifted_logits).sum())
        return shifted_logits - sumexp

    def convert_logits_normalization(
        self,
        current: LogitsNormalization,
        target: LogitsNormalization,
        logits: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
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
        self, logits: np.ndarray, indices: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        top_p_selection = min(logits.shape[-1], TOP_P_DEFAULT_SELECTION)
        tokens, values = self.sampler.select_top_k(logits, indices, -top_p_selection)
        probs = self._to_softmax(values, self.decode_config.logits_normalization)
        if probs is None:
            probs = np.array([])
        if indices is None and hasattr(probs, "__len__") and len(probs) > 0:
            sorted_order = np.argsort(probs)[::-1]
            tokens = tokens[sorted_order]
            probs = probs[sorted_order]
        return tokens, probs

    def _to_softmax(
        self, values: np.ndarray, logits_normalization: LogitsNormalization
    ) -> np.ndarray:
        if values is None or not hasattr(values, "__len__") or len(values) == 0:
            return np.array([])
        probs = self.convert_logits_normalization(
            logits_normalization, LogitsNormalization.SOFTMAX, values
        )
        if probs is None:
            probs = np.array([])
        return probs

    def _sample_logits_top_k(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        top_k: int,
        num_selections: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        tokens, values = self.sampler.select_top_k(logits, indices, -top_k)
        probs = self._to_softmax(values, self.decode_config.logits_normalization)
        if probs is None:
            probs = np.array([])
        if indices is None and hasattr(probs, "__len__") and len(probs) > 0:
            sorted_order = np.argsort(probs)[::-1]
            tokens = tokens[sorted_order]
            probs = probs[sorted_order]
        return self.sampler.sample_top_k(tokens=tokens, probs=probs, k=num_selections)

    def _sample_logits_top_p(
        self,
        tokens: np.ndarray,
        probs: np.ndarray,
        top_p: float,
        num_selections: int,
        return_probs: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if probs is None:
            probs = np.array([])
        config = self.decode_config
        if config.top_k is None:
            tokens, probs = self._pre_select_top_p(tokens, probs)
        if probs is None:
            probs = np.array([])
        result = self.sampler.sample_top_p(
            tokens=tokens,
            probs=probs,
            p=top_p,
            k=num_selections,
            return_probs=return_probs,
        )
        if result is None or len(result) != 2:
            return np.array([]), np.array([])
        return result


class BeamSearchBeam(BaseBeam):
    def _convert_results_to_log_probs(self, probs: np.ndarray) -> List[float]:
        if probs is None or len(probs) == 0:
            return []
        log_probs = self.convert_logits_normalization(
            LogitsNormalization.SOFTMAX, LogitsNormalization.LOG_SOFTMAX, probs
        )
        return log_probs.tolist()

    def sample_default(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        )
        return tokens, np.array(log_probs)

    def sample_top_k(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        top_k: int,
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._sample_logits_top_k(
            logits,
            indices,
            top_k,
            num_selections=self.decode_config.num_beams - num_completed_beams,
        )

    def sample_top_p(
        self,
        tokens: np.ndarray,
        probs: np.ndarray,
        top_p: float,
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._sample_logits_top_p(
            tokens,
            probs,
            top_p,
            num_selections=self.decode_config.num_beams - num_completed_beams,
            return_probs=True,
        )

    def get_results(
        self, tokens: np.ndarray, probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        log_probs = self._convert_results_to_log_probs(probs)
        return tokens, np.array(log_probs)


class DefaultBeam(BaseBeam):
    def sample_default(self, logits: np.ndarray, indices: Optional[np.ndarray], _):
        if indices is not None:
            if isinstance(indices, np.ndarray):
                return indices[0]
            elif hasattr(indices, "items"):
                return indices.items[0]
        return self.sampler.select_greedy(logits)

    def sample_top_k(
        self, logits: np.ndarray, indices: Optional[np.ndarray], top_k: int, _
    ):
        decode_config = self.decode_config
        num_selections = 1 if decode_config.top_p is None else top_k
        return self._sample_logits_top_k(
            logits, indices, top_k, num_selections=num_selections
        )

    def sample_top_p(self, tokens: np.ndarray, probs: np.ndarray, top_p: float, _):
        return self._sample_logits_top_p(tokens, probs, top_p, num_selections=1)

    def get_results(self, tokens: np.ndarray, _):
        return int(tokens[0])


def build_beam_group(
    exec_req: LlmInferenceExecRequest,
    config: TokenSelectionStrategyConfig,
    selection_callback: Callable[[List[BaseBeam], List[BaseBeam]], List[BaseBeam]],
) -> BeamGroup:
    decode_config = config.decode_config
    if not decode_config.use_beam_search and decode_config.num_beams > 1:
        exec_reqs = BaseBeam.replicate_inference_exec_requests(
            exec_req, decode_config.num_beams - 1
        )
    else:
        exec_reqs = [exec_req]
    beam_cls = BeamSearchBeam if decode_config.use_beam_search else DefaultBeam
    beams: List[BaseBeam] = [
        beam_cls(exec_req, decode_config=decode_config) for exec_req in exec_reqs
    ]
    return BeamGroup(
        config.eos_token_id, decode_config.num_beams, beams, selection_callback
    )


class BeamGroup:
    def __init__(
        self,
        eos_token_id: int,
        num_beams: int,
        beams: List[BaseBeam],
        selection_callback: Callable[[List[BaseBeam], List[BaseBeam]], List[BaseBeam]],
    ):
        self.beam_group_id = str(uuid4())
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.active_beams: List[BaseBeam] = beams
        self.selection_callback = selection_callback
        self.completed_beams: List[BaseBeam] = []

    @property
    def active_beam_count(self):
        return len(self.active_beams)

    async def wait(self):
        from asyncio import gather

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
        for beam in self.active_beams:
            if beam.exec_req not in active_reqs and beam.exec_req not in completed_reqs:
                beam.exec_req.free_cache_pages()
        self.active_beams = active_beams
        self.completed_beams.extend(completed_beams)

    def clean_up(self):
        logger.debug(f"Cleaning up {self.beam_group_id}...")
        for beam in self.active_beams + self.completed_beams:
            beam.exec_req.free_cache_pages()


# --- Scorers ---
class BaseBeamScorer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def update_score(self, beam: BaseBeam, value: float) -> None:
        pass

    @abstractmethod
    def finalize_score(self, beam: BaseBeam) -> None:
        pass

    @abstractmethod
    def normalize_score(self, beam: BaseBeam, value: float) -> float:
        pass

    @abstractmethod
    def score_beams(self, beams: List[BaseBeam]) -> List[BaseBeam]:
        pass

    @abstractmethod
    def select_beams(
        self, active_beams: List[BaseBeam], complete_beams: List[BaseBeam]
    ) -> List[BaseBeam]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def penalize_brevity(self, beam: BaseBeam) -> None:
        exec_req = beam.exec_req
        if len(exec_req.input_token_ids) - exec_req.prompt_length > 0:
            beam.score /= len(exec_req.input_token_ids) - exec_req.prompt_length
        return None


class DefaultScorer(BaseBeamScorer):
    def __init__(self, config):
        super().__init__(config)

    def update_score(self, beam: DefaultBeam, value: float) -> None:
        pass

    def finalize_score(self, beam: DefaultBeam) -> None:
        pass

    def normalize_score(self, beam: DefaultBeam, value: float) -> None:
        pass

    def score_beams(self, beams: List[DefaultBeam]) -> List[DefaultBeam]:
        return beams

    def select_beams(
        self, active_beams: List[DefaultBeam], completed_beams: List[DefaultBeam]
    ) -> List[DefaultBeam]:
        selections = []
        for beam in active_beams:
            if isinstance(beam, DefaultBeam):
                token = beam.sample_logits(len(completed_beams))
                if isinstance(token, int):
                    beam.last_token = token
            selections.append(beam)
        return selections

    def reset(self) -> None:
        pass


class BeamSearchScorer(BaseBeamScorer):
    def __init__(self, config):
        self.min_log_prob: float = 0.0
        self.top_score: Optional[float] = None
        self.top_beam: Optional[BeamSearchBeam] = None
        super().__init__(config)

    def update_score(self, beam: BeamSearchBeam, log_prob: float) -> None:
        if log_prob < self.min_log_prob:
            self.min_log_prob = log_prob
        beam.score += log_prob
        if self.top_score is None or beam.score > self.top_score:
            self.top_score = beam.score
            self.top_beam = beam

    def finalize_score(self, beam: BeamSearchBeam) -> None:
        beam.score = beam.score - beam.accumulated_normalization
        return self.penalize_brevity(beam)

    def normalize_score(self, beam: BeamSearchBeam, min_log_prob: float) -> None:
        beam.accumulated_normalization += abs(min_log_prob)

    def score_beams(
        self, beams: List[BeamSearchBeam], k: int, normalize: bool = True
    ) -> List[BeamSearchBeam]:
        beams = [b for b in beams if isinstance(b, BeamSearchBeam)]
        sorted_selections = sorted(beams, key=lambda beam: beam.score, reverse=True)[:k]
        if normalize:
            for beam in sorted_selections:
                self.normalize_score(beam, self.min_log_prob)
        return sorted_selections

    def select_beams(
        self, active_beams: List[BeamSearchBeam], completed_beams: List[BeamSearchBeam]
    ) -> List[BeamSearchBeam]:
        config = self.config
        num_beams = config.decode_config.num_beams
        k = num_beams - len(completed_beams)
        # Only operate on BeamSearchBeam objects
        active_beams = [b for b in active_beams if isinstance(b, BeamSearchBeam)]
        completed_beams = [b for b in completed_beams if isinstance(b, BeamSearchBeam)]
        selections: List[BeamSearchBeam] = []
        for beam in active_beams:
            top_tokens, top_values = beam.sample_logits(len(completed_beams))
            for token, value in zip(top_tokens, top_values):
                new_beam = BeamSearchBeam.clone(beam)
                new_beam.last_token = token
                if isinstance(new_beam, BeamSearchBeam):
                    self.update_score(new_beam, value)
                    selections.append(new_beam)
        if len(selections) < k and self.top_beam is not None:
            beams_to_add = num_beams - len(selections)
            for _ in range(beams_to_add):
                new_beam = BeamSearchBeam.clone(self.top_beam)
                selections.append(new_beam)
        selections = self.score_beams(selections, k, normalize=True)
        self.reset()
        return selections

    def reset(self):
        self.min_log_prob = 0.0
        self.top_score = None
        self.top_beam = None


# --- Token Selector ---
@dataclass
class TokenSelector(BaseTokenSelectionStrategy):
    scorer: Union[BeamSearchScorer, DefaultScorer]
    min_log_prob: float = 0.0

    def _stream_single_beam(self, beam_group: BeamGroup):
        results_callback = self.token_selection_strategy_config.results_callback
        assert (
            beam_group.num_beams == 1
        ), "Streaming is not supported for multi-hypothesis yet."
        beam = beam_group.active_beams[0]
        if isinstance(beam.last_token, int):
            results_callback(beam.last_token)
        elif isinstance(beam.last_token, list):
            results_callback(beam.last_token)

    async def decode(self, exec_req: LlmInferenceExecRequest):
        self._log_sampling_method()
        config = self.token_selection_strategy_config
        use_beam_search = config.decode_config.use_beam_search
        exec_req.reset(InferencePhase.DECODE)
        beam_group = build_beam_group(
            exec_req,
            config,
            cast(
                Callable[[List[BaseBeam], List[BaseBeam]], List[BaseBeam]],
                self.scorer.select_beams,
            ),
        )
        reservations = beam_group.active_beam_count
        config.decode_begin_callback(rid=exec_req.orig_instance_id, count=reservations)
        for _ in range(config.decode_config.max_completion_tokens):
            if (
                exec_req.status_tracker is not None
                and exec_req.status_tracker.is_disconnected()
            ):
                break
            active_beam_count = len(beam_group.active_beams)
            if reservations > active_beam_count:
                release_amount = reservations - active_beam_count
                config.decode_end_callback(
                    rid=exec_req.orig_instance_id, count=release_amount
                )
                reservations = active_beam_count
            if reservations < active_beam_count:
                acquire_amount = active_beam_count - reservations
                config.decode_begin_callback(
                    rid=exec_req.orig_instance_id, count=acquire_amount
                )
                reservations = active_beam_count
            for beam in beam_group.active_beams:
                req = beam.exec_req
                req.reset(InferencePhase.DECODE)
                config.decode_callback(req)
            await beam_group.wait()
            beam_group.process_beams()
            if not beam_group.active_beams:
                break
            if config.decode_config.num_beams == 1 and not use_beam_search:
                self._stream_single_beam(beam_group)
        config.decode_end_callback(rid=exec_req.orig_instance_id, count=reservations)
        beam_group.clean_up()
        self.get_results(beam_group)

    def _get_results_beam_search(self, beam_group: BeamGroup, results: List[List[int]]):
        for beam in beam_group.active_beams:
            self.scorer.finalize_score(beam)
        active_beams = [
            b for b in beam_group.active_beams if isinstance(b, BeamSearchBeam)
        ]
        active_beams = self.scorer.score_beams(
            active_beams, len(active_beams), normalize=False
        )
        for i in range(beam_group.num_beams - len(results)):
            beam = active_beams[i]
            results.append(beam.exec_req.input_token_ids[beam.exec_req.prompt_length :])
        return results

    def get_results(self, beam_group: BeamGroup):
        config = self.token_selection_strategy_config
        use_beam_search = config.decode_config.use_beam_search
        if config.decode_config.num_beams == 1 and not use_beam_search:
            return
        results = [
            beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
            for beam in beam_group.completed_beams
        ]
        if len(results) < beam_group.num_beams:
            if use_beam_search:
                results = self._get_results_beam_search(beam_group, results)
            else:
                results.extend(
                    [
                        beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
                        for beam in beam_group.active_beams
                    ]
                )
        if isinstance(results, list):
            config.results_callback(results)
        else:
            config.results_callback([results])


# --- API Builders ---
def build_token_selector_config(
    decode_config: DecodeConfig,
    prefill_batcher,
    decode_batcher,
    results_callback: Callable[[Union[int, List[int]]], None],
    eos_token_id: int,
) -> TokenSelectionStrategyConfig:
    return TokenSelectionStrategyConfig(
        decode_config,
        prefill_callback=prefill_batcher.submit,
        decode_callback=decode_batcher.submit,
        decode_begin_callback=decode_batcher.reserve_workitem,
        decode_end_callback=decode_batcher.complete_workitem,
        results_callback=results_callback,
        eos_token_id=eos_token_id,
    )


def build_token_selector(
    config: TokenSelectionStrategyConfig,
) -> BaseTokenSelectionStrategy:
    scorer = (
        BeamSearchScorer(config=config)
        if config.decode_config.use_beam_search
        else DefaultScorer(config=config)
    )
    return TokenSelector(token_selection_strategy_config=config, scorer=scorer)


def is_multi_response(decode_config: DecodeConfig) -> bool:
    use_beam_search = decode_config.use_beam_search
    num_beams = decode_config.num_beams
    return use_beam_search or num_beams > 1


__all__ = [
    "build_token_selector",
    "build_token_selector_config",
    "BaseTokenSelectionStrategy",
    "BeamSearchScorer",
    "DefaultScorer",
    "get_strategy_from_str",
    "is_multi_response",
    "Sampler",
    "TokenSelectionStrategyConfig",
    "TokenSelectionStrategy",
    "TokenSelector",
]
