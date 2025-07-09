# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from asyncio import gather
from typing import Dict, List, Set, Callable, Union, Sequence
from uuid import uuid4
import logging

from ..messages import LlmInferenceExecRequest, InferencePhase
from .config import DecodeConfig, TokenSelectionStrategyConfig

logger = logging.getLogger(__name__)


class BaseBeam(ABC):
    def __init__(
        self,
        exec_req: LlmInferenceExecRequest,
        decode_config: DecodeConfig,
    ):
        self.exec_req = exec_req
        self.decode_config = decode_config
        self.score = 0.0
        self.last_token = -1

    @staticmethod
    def replicate_inference_exec_requests(
        exec_req: LlmInferenceExecRequest,
        beam_count: int,
    ) -> List[LlmInferenceExecRequest]:
        requests = [exec_req]

        for i in range(beam_count):
            copied_exec_req = LlmInferenceExecRequest.copy_exec_request(exec_req)
            requests.append(copied_exec_req)

        return requests

    @abstractmethod
    def sample_logits(self, completed_beams_count: int) -> int:
        """Sample a token from the logits."""
        pass


class DefaultBeam(BaseBeam):
    def __init__(
        self,
        exec_req: LlmInferenceExecRequest,
        decode_config: DecodeConfig,
    ):
        super().__init__(exec_req, decode_config)

    def sample_logits(self, completed_beams_count: int) -> int:
        # Simple greedy sampling for default beam with some diversity
        import numpy as np

        if self.exec_req.result_logits is not None:
            logits = np.array(self.exec_req.result_logits.items)

            # For multi-beam scenarios, add some diversity by considering top-k tokens
            if (
                hasattr(self.decode_config, "num_beams")
                and self.decode_config.num_beams > 1
            ):
                # Use a simple strategy: pick from top-k tokens to add diversity
                k = min(5, len(logits))  # Consider top-5 tokens
                top_k_indices = np.argpartition(logits, -k)[-k:]
                # Use modulo to select different tokens for different beams
                beam_offset = (
                    completed_beams_count + hash(str(self.exec_req.instance_id))
                ) % k
                selected_idx = top_k_indices[
                    np.argsort(logits[top_k_indices])[-1 - beam_offset]
                ]
                token = int(selected_idx)
            else:
                token = int(np.argmax(logits))
            return token
        return 0


class BeamSearchBeam(BaseBeam):
    def __init__(
        self,
        exec_req: LlmInferenceExecRequest,
        decode_config: DecodeConfig,
    ):
        super().__init__(exec_req, decode_config)
        self.accumulated_normalization = 0.0

    def sample_logits(self, completed_beams_count: int) -> int:
        # Beam search sampling with diversity
        import numpy as np

        if self.exec_req.result_logits is not None:
            logits = np.array(self.exec_req.result_logits.items)
            # Apply temperature if needed
            if (
                hasattr(self.decode_config, "temperature")
                and self.decode_config.temperature != 1.0
            ):
                logits = logits / self.decode_config.temperature

            # For beam search, select from top-k candidates to maintain diversity
            num_beams = getattr(self.decode_config, "num_beams", 1)
            if num_beams > 1:
                # Use beam search logic: consider multiple top candidates
                k = min(
                    num_beams * 2, len(logits)
                )  # Consider 2x beams worth of candidates
                top_k_indices = np.argpartition(logits, -k)[-k:]
                top_k_logits = logits[top_k_indices]

                # Sort by logit values (highest first)
                sorted_indices = top_k_indices[np.argsort(top_k_logits)[::-1]]

                # Select different tokens for different beams
                beam_index = (
                    completed_beams_count + hash(str(self.exec_req.instance_id))
                ) % len(sorted_indices)
                token = int(sorted_indices[beam_index])
            else:
                token = int(np.argmax(logits))
            return token
        return 0


def build_beam_group(
    exec_req: LlmInferenceExecRequest,
    config: TokenSelectionStrategyConfig,
    selection_callback: Callable,
) -> "BeamGroup":
    """Select the appropriate beam class based on the decode configuration."""
    decode_config = config.decode_config

    # Create beam requests - always replicate for multiple beams
    if decode_config.num_beams > 1:
        exec_reqs = BaseBeam.replicate_inference_exec_requests(
            exec_req,
            decode_config.num_beams - 1,
        )
    else:
        exec_reqs = [exec_req]

    # Create beams
    beam_cls = BeamSearchBeam if decode_config.use_beam_search else DefaultBeam
    beams = [beam_cls(req, decode_config=decode_config) for req in exec_reqs]

    return BeamGroup(
        decode_config.eos_token_id,
        decode_config.num_beams,
        beams,
        selection_callback,
    )


class BeamGroup:
    def __init__(
        self,
        eos_token_id: int,
        num_beams: int,
        beams: Sequence[BaseBeam],
        selection_callback: Callable[
            [List[BaseBeam], List[BaseBeam]],
            List[BaseBeam],
        ],
    ):
        self.beam_group_id = str(uuid4())
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.active_beams = list(beams)
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

        logger.debug(
            f"Processing {len(beam_selections)} beam selections, EOS token ID: {self.eos_token_id}"
        )

        for i in range(len(beam_selections)):
            beam = beam_selections[i]
            new_req, token = beam.exec_req, beam.last_token

            if new_req.instance_id in visited_reqs:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = LlmInferenceExecRequest.copy_exec_request(visited_req)
                beam.exec_req = new_req

            visited_reqs[new_req.instance_id] = new_req

            logger.debug(
                f"Beam {i}: token={token}, eos_token_id={self.eos_token_id}, is_eos={token == self.eos_token_id}"
            )

            if token == self.eos_token_id:
                completed_beams.append(beam)
                completed_reqs.add(new_req)
                logger.debug(f"Beam {i} completed with EOS token {token}")
            else:
                active_beams.append(beam)
                active_reqs.add(new_req)
                logger.debug(f"Beam {i} continues with token {token}")

        # Update beam states
        all_beams_to_update = completed_beams + active_beams
        for beam in all_beams_to_update:
            if beam.last_token != -1:
                beam.exec_req.input_token_ids.append(beam.last_token)

        # Free cache pages for completed requests
        for req in completed_reqs:
            req.free_cache_pages()

        # Clean up unused requests
        for beam in self.active_beams:
            req = beam.exec_req
            if req not in active_reqs and req not in completed_reqs:
                req.free_cache_pages()

        self.active_beams = active_beams
        self.completed_beams.extend(completed_beams)

    def clean_up(self):
        logger.debug(f"Cleaning up BeamGroup {self.beam_group_id}...")
        all_beams = list(self.active_beams) + list(self.completed_beams)
        for beam in all_beams:
            beam.exec_req.free_cache_pages()
