import logging
import numpy as np
from typing import Optional, Tuple, Union

import shortfin.array as sfnp

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import DecodeConfig, LogitsNormalization
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
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the logits using the default sampling strategy for a give `Beam`.

        Args:
            logits (np.ndarray): The logits from which to select.
            indices (np.ndarray | None): Optional pre-selected indices.
        """

    @abstractmethod
    def sample_top_k(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        top_k: int,
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the top-k tokens from the logits.

        Args:
            logits (np.ndarray): The logits from which to select.
            indices (np.ndarray | None): Optional pre-selected indices.
            top_k (int): The number of top tokens to select.
            num_completed_beams (int): Number of completed beams.
        """

    @abstractmethod
    def sample_top_p(
        self,
        tokens: np.ndarray,
        probs: np.ndarray,
        top_p: float,
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the top-p tokens from the logits.

        Args:
            tokens (np.ndarray): The tokens to sample from.
            probs (np.ndarray): The probabilities of the tokens.
            top_p (float): The cumulative probability threshold for sampling.
            num_completed_beams (int): Number of completed beams.
        """

    @abstractmethod
    def get_results(
        self, tokens: np.ndarray, probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the results to a format suitable for the beam.

        Args:
            tokens (np.ndarray): The selected tokens.
            probs (np.ndarray): The probabilities of the selected tokens.
        """

    @classmethod
    def clone(cls, beam: "BaseBeam") -> "BaseBeam":
        return cls(
            exec_req=beam.exec_req,
            score=beam.score,
            accumulated_normalization=beam.accumulated_normalization,
            last_token=beam.last_token,
            decode_config=beam.decode_config,
        )

    def sample_logits(self, num_completed_beams: int) -> Tuple[np.ndarray, np.ndarray]:
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

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature to the logits of a decode invocation.

        Args:
            temperature (float): Value to use for `temperature`.
        """
        if self.decode_config.temperature == 1.0:
            return logits
        return np.divide(logits, self.decode_config.temperature)

    def _softmax(self, logits: Union[np.ndarray, sfnp.device_array]) -> np.ndarray:
        if isinstance(logits, sfnp.device_array):
            logits = np.array(logits)

        x_max = np.max(logits)
        e_x = np.exp(logits - x_max)
        return e_x / np.sum(e_x)

    def _log_softmax(self, logits: Union[np.ndarray, sfnp.device_array]) -> np.ndarray:
        if isinstance(logits, sfnp.device_array):
            logits = np.array(logits)

        c = logits.max()
        shifted_logits = logits - c
        sumexp = np.log(np.exp(shifted_logits).sum())
        return shifted_logits - sumexp

    def _convert_logits_normalization(
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
        values: np.ndarray,
        logits_normalization: LogitsNormalization,
    ) -> np.ndarray:
        return self._convert_logits_normalization(
            logits_normalization,
            LogitsNormalization.SOFTMAX,
            values,
        )

    def _sample_logits_top_k(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        top_k: int,
        num_selections: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        self,
        tokens: np.ndarray,
        probs: np.ndarray,
        top_p: float,
        num_selections: int,
        return_probs: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        probs: np.ndarray,
    ) -> np.ndarray:
        log_probs = self._convert_logits_normalization(
            LogitsNormalization.SOFTMAX,
            LogitsNormalization.LOG_SOFTMAX,
            probs,
        )

        return log_probs

    def sample_default(
        self,
        logits: np.ndarray,
        indices: Optional[np.ndarray],
        num_completed_beams: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        log_probs = self._convert_logits_normalization(
            self.decode_config.logits_normalization,
            LogitsNormalization.LOG_SOFTMAX,
            probs,
        ).tolist()

        return tokens, log_probs

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
        self, tokens: np.ndarray, probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the results to log probabilities for beam search.

        Args:
            tokens (np.ndarray): The selected tokens.
            probs (np.ndarray): The probabilities of the selected tokens.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The tokens and their log probabilities.
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
        tokens: np.ndarray,
        probs: np.ndarray,
        top_p: float,
        _,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the top-p tokens from the logits.

        Args:
            tokens (np.ndarray): The tokens to sample from.
            probs (np.ndarray): The probabilities of the tokens.
            top_p (float): The cumulative probability threshold for sampling.
            num_completed_beams (int): Number of completed beams.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The selected tokens and their probabilities.
        """
        return self._sample_logits_top_p(
            tokens,
            probs,
            top_p,
            num_selections=1,
        )

    def get_results(self, tokens: np.ndarray, _: np.ndarray) -> int:
        return int(tokens[0])
