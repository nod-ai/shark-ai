from abc import ABC, abstractmethod
from typing import List

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
    def score_beams(beams: List[BaseBeam]) -> List[BaseBeam]:
        """Score a group of beams.

        Args:
            beams (List[BaseBeam]): The beams to score.

        Returns:
            List[BaseBeam]: The scored beams in descending order of score.
        """

    @abstractmethod
    def select_beams(
        active_beams: List[BaseBeam], complete_beams: List[BaseBeam]
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

    def score_beams(self, beams: List[DefaultBeam]) -> List[DefaultBeam]:
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
            active_beams (List[IndependentBeam]): Beams that are still active.
            completed_beams (Set[IndependentBeam]): Beams that have been completed.

        Returns:
            List[IndependentBeam]: The `top_k` selections, containing necessary info for `beam_group` to handle choosing and processing beams.
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
                new_beam = BeamSearchBeam.clone(self.scorer.top_beam)
                selections.append(new_beam)

        selections = self.score_beams(selections, k)
        self.reset()
        return selections

    def reset(self):
        """Reset the scorer state."""
        self.min_log_prob = 0.0
        self.top_score = None
