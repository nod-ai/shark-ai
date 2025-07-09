# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import List, Callable
import shortfin as sf

from .beam_group import BaseBeam, DefaultBeam, BeamSearchBeam


class BaseBeamScorer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def update_score(
        self,
        beam: BaseBeam,
        value: float,
    ) -> None:
        """Update the score of a `beam`.

        Args:
            beam (BaseBeam): The beam to update.
            value (float): Value to update the score with.
        """

    @abstractmethod
    def finalize_score(self, beam: BaseBeam) -> None:
        """Define a `final_score` for a given beam, if applicable.

        Args:
            beam (BaseBeam): The beam to update.
        """

    @abstractmethod
    def normalize_score(
        self,
        beam: BaseBeam,
        value: float,
    ) -> float:
        """Normalize the score of a `beam`.

        Args:
            beam (BaseBeam): The beam to normalize.
            value (float): Value to normalize the score with.

        Returns:
            float: Normalized score.
        """

    @abstractmethod
    def score_beams(
        self, beams: List[BaseBeam], k: int, normalize: bool
    ) -> List[BaseBeam]:
        """Score a group of beams.

        Args:
            beams (List[BaseBeam]): The beams to score.

        Returns:
            List[BaseBeam]: The scored beams in descending order of score.
        """

    @abstractmethod
    def select_beams(
        self, active_beams: List[BaseBeam], complete_beams: List[BaseBeam]
    ) -> List[BaseBeam]:
        """Select the next candidate set of beams for decode invocation.

        Args:
            active_beams (List[BaseBeam]): The beams still actively being decoded.
            complete_beams (List[BaseBeam]): The beams that are completed.

        Returns:
            List[BaseBeam]: Selected beams.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the state of the scorer.

        This is useful when reusing the scorer for multiple decoding iterations.
        """

    def create_cpp_selection_callback(self):
        """Create a selection callback compatible with C++ BeamProcessor.

        This method converts the Python beam scorer logic into a callback function
        that can be used with the C++ BeamProcessor implementation.

        Returns:
            Callable: A function that takes (active_beams, completed_beams) and returns selected beams
        """

        def selection_callback(cpp_active_beams, cpp_completed_beams):
            # Convert C++ BeamState objects to Python BaseBeam objects
            py_active_beams = []
            for cpp_beam in cpp_active_beams:
                py_beam = self._convert_cpp_beam_to_python(cpp_beam)
                py_active_beams.append(py_beam)

            py_completed_beams = []
            for cpp_beam in cpp_completed_beams:
                py_beam = self._convert_cpp_beam_to_python(cpp_beam)
                py_completed_beams.append(py_beam)

            # Use the existing select_beams logic
            selected_py_beams = self.select_beams(py_active_beams, py_completed_beams)

            # Convert selected beams back to C++ BeamState objects
            selected_cpp_beams = []
            for py_beam in selected_py_beams:
                cpp_beam = self._convert_python_beam_to_cpp(py_beam)
                selected_cpp_beams.append(cpp_beam)

            return selected_cpp_beams

        return selection_callback

    def _convert_cpp_beam_to_python(self, cpp_beam: sf.llm.BeamState) -> BaseBeam:
        """Convert C++ BeamState to Python BaseBeam."""
        # Create a mock Python execution request from C++ data
        from ..messages import LlmInferenceExecRequest

        # Create a minimal exec_req that matches the interface expected by BaseBeam
        class MockExecRequest:
            def __init__(self, cpp_req):
                self.instance_id = cpp_req.instance_id
                self.input_token_ids = cpp_req.input_token_ids.copy()
                self.start_position = cpp_req.start_position
                self.prompt_length = cpp_req.prompt_length
                self.result_logits = None  # Will be populated when needed
                self.result_indices = None

            def copy_exec_request(self):
                copy = MockExecRequest(self)
                return copy

        mock_exec_req = MockExecRequest(cpp_beam.exec_req)

        # Create the appropriate beam type based on config
        if (
            hasattr(self.config, "decode_config")
            and self.config.decode_config.use_beam_search
        ):
            py_beam = BeamSearchBeam(
                mock_exec_req, decode_config=self.config.decode_config
            )
        else:
            py_beam = DefaultBeam(
                mock_exec_req, decode_config=self.config.decode_config
            )

        # Copy beam state
        py_beam.score = cpp_beam.score
        py_beam.accumulated_normalization = cpp_beam.accumulated_normalization
        py_beam.last_token = cpp_beam.last_token

        return py_beam

    def _convert_python_beam_to_cpp(self, py_beam: BaseBeam) -> sf.llm.BeamState:
        """Convert Python BaseBeam to C++ BeamState."""
        # Create C++ InferenceRequest
        cpp_req = sf.llm.InferenceRequest()
        cpp_req.instance_id = py_beam.exec_req.instance_id
        cpp_req.input_token_ids = py_beam.exec_req.input_token_ids.copy()
        cpp_req.start_position = py_beam.exec_req.start_position
        cpp_req.prompt_length = py_beam.exec_req.prompt_length

        # Create C++ BeamState
        cpp_beam = sf.llm.BeamState(cpp_req)
        cpp_beam.score = py_beam.score
        cpp_beam.accumulated_normalization = py_beam.accumulated_normalization
        cpp_beam.last_token = py_beam.last_token

        return cpp_beam

    def penalize_brevity(
        self,
        beam: BaseBeam,
    ) -> float:
        """Apply a length penalty to the score of a `beam`.

        Args:
            beam (BaseBeam): The beam to penalize.
            length (int): Length of the sequence.

        Returns:
            float: Penalized score.
        """
        # TODO(stbaione): Extend this to support other length penalty types
        exec_req = beam.exec_req
        beam.score /= len(exec_req.input_token_ids) - exec_req.prompt_length


class DefaultScorer(BaseBeamScorer):
    def __init__(self, config):
        super().__init__(config)

    def update_score(self, beam: DefaultBeam, value: float) -> None:
        pass

    def finalize_score(self, beam: DefaultBeam) -> None:
        pass

    def normalize_score(self, beam: DefaultBeam, value: float) -> None:
        pass

    def score_beams(
        self, beams: List[DefaultBeam], k: int, normalize: bool
    ) -> List[DefaultBeam]:
        return beams

    def select_beams(
        self, active_beams: List[DefaultBeam], completed_beams: List[DefaultBeam]
    ) -> List[DefaultBeam]:
        """Select the next candidate set of beams for decode invocation.

        Args:
            active_beams (List[DefaultBeam]): The beams still actively being decoded.
            completed_beams (List[DefaultBeam]): The beams that are completed.

        Returns:
            List[DefaultBeam]: Selected beams.
        """
        selections = []

        # Sample logits for each active beam for it to select its next token.
        for beam in active_beams:
            token = beam.sample_logits(len(completed_beams))
            beam.last_token = token
            selections.append(
                beam,
            )

        return selections

    def reset(self) -> None:
        """Reset the state of the scorer."""
        pass

    def penalize_brevity(self, beam):
        pass


class BeamSearchScorer(BaseBeamScorer):
    def __init__(self, config):
        self.min_log_prob: float = 0.0
        self.top_score: float | None = None
        self.top_beam: BeamSearchBeam | None = None

        super().__init__(config)

    def update_score(
        self,
        beam: BeamSearchBeam,
        log_prob: float,
    ) -> None:
        """Update the score of a beam with the log probability of the selected token.

        Args:
            beam (BeamSearchBeam): The beam to update.
            log_prob (float): Log probability of the token.
        """
        if log_prob < self.min_log_prob:
            self.min_log_prob = log_prob

        beam.score += log_prob

        if self.top_score is None or beam.score > self.top_score:
            self.top_score = beam.score
            self.top_beam = beam

    def finalize_score(
        self,
        beam: BeamSearchBeam,
    ) -> None:
        """Finalize the score of a beam after all tokens have been selected.

        Args:
            beam (BeamSearchBeam): The beam to finalize.
        """
        beam.score = beam.score - beam.accumulated_normalization
        return self.penalize_brevity(beam)

    def normalize_score(
        self,
        beam: BeamSearchBeam,
        min_log_prob: float,
    ) -> None:
        """Normalize the score of a beam based on the minimum log probability.

        Args:
            beam (BeamSearchBeam): The beam to normalize.
            min_log_prob (float): Minimum log probability of the selected tokens.
        """
        beam.accumulated_normalization += abs(min_log_prob)

    def score_beams(self, beams, k: int, normalize: bool = True):
        sorted_selections = sorted(beams, key=lambda beam: beam.score, reverse=True)[:k]
        if normalize:
            for beam in sorted_selections:
                self.normalize_score(beam, self.min_log_prob)

        return sorted_selections

    def select_beams(
        self,
        active_beams: List[BeamSearchBeam],
        completed_beams: List[BeamSearchBeam],
    ) -> List[BeamSearchBeam]:
        """Handle the selection of the `top_k` beams within a decode step.

        Args:
            active_beams (List[BeamSearchBeam]): Beams that are still active.
            completed_beams (List[BeamSearchBeam]): Beams that have been completed.

        Returns:
            List[BeamSearchBeam]: The `top_k` selections, containing necessary info for `beam_group` to handle choosing and processing beams.
        """
        config = self.config
        num_beams = config.decode_config.num_beams
        k = num_beams - len(completed_beams)
        selections: List[BeamSearchBeam] = []

        # Parse each beam to select the next candidates
        for beam in active_beams:
            top_tokens, top_values = beam.sample_logits(len(completed_beams))
            for token, value in zip(top_tokens, top_values):

                new_beam = BeamSearchBeam.clone(beam)
                new_beam.last_token = token
                self.update_score(new_beam, value)
                selections.append(new_beam)

        # Ensure we have enough beams to fill the `num_beams` requirement
        if len(selections) < k:
            beams_to_add = num_beams - len(selections)
            for _ in range(beams_to_add):
                new_beam = BeamSearchBeam.clone(self.top_beam)
                selections.append(new_beam)

        selections = self.score_beams(selections, k)
        self.reset()
        return selections

    def reset(self):
        """Reset the scorer state."""
        self.min_log_prob = 0.0
        self.top_score = None
