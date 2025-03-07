from .base_decode_strategy import DecodeStrategy, DecodeStrategyConfig


from asyncio import gather
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple
from uuid import uuid4

import numpy as np

from ..messages import LlmInferenceExecRequest, InferencePhase


logger = logging.getLogger(__name__)


@dataclass
class ExecRequestSelection:
    """Helper class top make `BeamGroup.evaluate_top_k` return cleaner."""

    log_prob: float
    exec_req: LlmInferenceExecRequest
    token: int
    min_log_prob: float


class BeamGroup:
    def __init__(
        self,
        beam_group_id: str,
        n_beams: int,
        temperature: int,
        exec_reqs: list[LlmInferenceExecRequest],
    ):
        self.beam_group_id = beam_group_id
        self.n_beams = n_beams
        self.temperature = temperature
        self.exec_reqs = exec_reqs
        self.completed_reqs: set[LlmInferenceExecRequest] = set()

    async def wait(self):
        done_signals = [
            req.done for req in self.exec_reqs if req not in self.completed_reqs
        ]
        return await gather(*done_signals)

    def _apply_temperature(self, logits: np.array) -> np.array:
        if self.temperature != 1.0:
            return logits / self.temperature

        return logits

    def log_softmax(self, logits: np.array) -> np.array:
        return logits
        # TODO: Move this to sfnp.array
        c = logits.max()
        logsumexp = np.log(np.exp(logits - c).sum())
        return logits - c - logsumexp

    def topk(
        self, logits: np.array, k: int, axis: int
    ) -> Tuple[List[float], List[int]]:
        # TODO: Move this to sfnp.array
        # indices = np.argpartition(logits, -k, axis=axis)
        # topk_indices = indices[axis][-k:]
        topk_indices = np.arange(k)
        topk_values = logits[axis][topk_indices]

        return topk_values, topk_indices

    def _get_exec_req_selections(
        self,
        log_prob_map: Dict[float, tuple[LlmInferenceExecRequest, int]],
        min_log_prob: int,
    ):
        # Find the topk tokens across all exec_reqs
        sorted_keys = sorted(log_prob_map.keys(), reverse=True)
        exec_req_selections: List[ExecRequestSelection] = []

        desired_selections = self.n_beams - len(self.completed_reqs)
        available_keys = len(sorted_keys)

        logger.info(
            f"Selecting top {desired_selections} from {available_keys} candidates (completed_reqs: {len(self.completed_reqs)})"
        )

        # get to the full number of beams (n_beams) to fill the cache; it's possible to have fewer beams due to key collisions in log_prob_map
        if available_keys < desired_selections:
            logger.warning(
                f"Not enough candidates ({available_keys}) to fill desired beam count ({desired_selections}). Duplicating best candidates."
            )

            for key in sorted_keys:
                exec_req, token = log_prob_map[key]
                exec_req_selections.append(
                    ExecRequestSelection(
                        log_prob=key - min_log_prob,
                        exec_req=exec_req,
                        token=token,
                        min_log_prob=min_log_prob,
                    )
                )

            # Duplicate best to fill
            remaining = desired_selections - available_keys
            for i in range(remaining):
                key = sorted_keys[i % available_keys]
                exec_req, token = log_prob_map[key]

                # Create a small offset to make log_prob slightly different to ensures they're treated as distinct paths in the beam search
                log_prob_offset = (i + 1) * 0.00001  # Very small offset

                exec_req_selections.append(
                    ExecRequestSelection(
                        log_prob=key - min_log_prob - log_prob_offset,
                        exec_req=exec_req,
                        token=token,
                        min_log_prob=min_log_prob,
                    )
                )
        else:
            # We have enough candidates, just take the top desired_selections
            for key in sorted_keys[:desired_selections]:
                exec_req, token = log_prob_map[key]
                exec_req_selections.append(
                    ExecRequestSelection(
                        # Shift log_probs to the right to avoid large
                        # negative numbers
                        log_prob=key - min_log_prob,
                        exec_req=exec_req,
                        token=token,
                        min_log_prob=min_log_prob,
                    )
                )

        return exec_req_selections

    def evaluate_topk(self) -> List[ExecRequestSelection]:
        # TODO: Use temperature when processing logits for better diversity of
        # outputs.
        exec_reqs = self.exec_reqs

        # Log how many active and completed beams we have
        active_reqs = [req for req in exec_reqs if req not in self.completed_reqs]
        logger.info(
            f"Evaluating topk with {len(active_reqs)} active beams out of {len(exec_reqs)} total (completed: {len(self.completed_reqs)})"
        )

        log_prob_map: Dict[float, tuple[LlmInferenceExecRequest, int]] = {}
        global_min_log_prob = 0.0
        # Find the topk tokens for each req in our beam group
        for exec_req in exec_reqs:
            if exec_req in self.completed_reqs:
                continue
            # NOTE: This copy is slow, and part of why this needs to be moved to
            # `shortfin.array`
            logits = np.array(exec_req.result_logits)
            # logits = self._apply_temperature(logits)

            # Take log_softmax. This is to avoid a req's cumulative probability
            # becoming too small, which can lead precision issues.
            # This allows us to obtain cumulative probability by summing
            # the log_probabilities, instead of multiplying the probabilities.
            log_logits = self.log_softmax(logits)
            log_logits = np.squeeze(log_logits, 1)
            values, tokens = self.topk(log_logits, self.n_beams, -1)
            min_log_prob = 0.0
            for value, token in zip(values, tokens):
                if value < min_log_prob:
                    min_log_prob = value
                cumulative_log_prob = exec_req.cumulative_log_prob + value
                log_prob_map[cumulative_log_prob] = (exec_req, token)

            if min_log_prob < global_min_log_prob:
                global_min_log_prob = min_log_prob

        return self._get_exec_req_selections(log_prob_map, global_min_log_prob)

    def process_beams(self, eos_token_id):
        exec_reqs_selections = self.evaluate_topk()
        visited_reqs: Dict[str, LlmInferenceExecRequest] = {}
        new_reqs = set()

        # Keep track of how many beam selections we're processing
        logger.info(
            f"BeamGroup processing {len(exec_reqs_selections)} beam selections out of {self.n_beams} total beams"
        )
        logger.info(f"Current active exec_reqs count: {len(self.exec_reqs)}")
        logger.info(f"Current completed_reqs count: {len(self.completed_reqs)}")

        for selection in exec_reqs_selections:
            new_req = selection.exec_req
            token = selection.token
            if new_req.instance_id not in visited_reqs:
                new_req.input_token_ids.append(token)
                new_req.output_token_ids.append(token)
                new_req.start_position += 1
                new_req.accumulated_normalization += abs(selection.min_log_prob)

            else:
                visited_req = visited_reqs[new_req.instance_id]
                new_req = visited_req.replicate_self()
                new_req.input_token_ids.append(token)
                new_req.output_token_ids.append(token)

            new_req.cumulative_log_prob = selection.log_prob
            visited_reqs[new_req.instance_id] = new_req
            new_reqs.add(new_req)
            # if token == eos_token_id:
            if False:
                new_req.llm_inference_metrics.set_end_time()
                self.completed_reqs.add(new_req)

        for req in self.exec_reqs:
            if req not in new_reqs:
                logger.info(
                    f"Freeing cache pages for req that didn't make it to new_reqs: {req.instance_id}"
                )
                req.free_cache_pages()

        for req in self.completed_reqs:
            logger.info(f"Freeing cache pages for completed req: {req.instance_id}")
            req.free_cache_pages()

        self.exec_reqs = list(new_reqs)
        logger.info(f"After processing, new exec_reqs count: {len(self.exec_reqs)}")

    def _final_score(self, exec_req: LlmInferenceExecRequest):
        return (
            exec_req.cumulative_log_prob - exec_req.accumulated_normalization
        ) / len(exec_req.output_token_ids)

    def find_top_beam(self) -> LlmInferenceExecRequest:
        completed_reqs = list(self.completed_reqs)
        if not completed_reqs:
            completed_reqs = self.exec_reqs
        max_score = self._final_score(completed_reqs[0])
        selected_req = completed_reqs[0]
        for req in completed_reqs[1:]:
            score = self._final_score(req)
            if score > max_score:
                selected_req = req
                max_score = score

        return selected_req

    def clean_up(self):
        logger.info("Cleaning up...")
        for req in self.exec_reqs:
            req.free_cache_pages()

        for req in self.completed_reqs:
            req.free_cache_pages()


@dataclass
class BeamSearchDecodeStrategyConfig(DecodeStrategyConfig):
    n_beams: int
    temperature: int
    return_top_k: bool = False


class BeamSearchDecodeStrategy(DecodeStrategy):
    beam_map: dict[str, BeamGroup] = {}

    def __init__(
        self,
        decode_strategy_config: BeamSearchDecodeStrategyConfig,
    ):
        self._decode_strategy_config = decode_strategy_config

    @property
    def decode_strategy_config(self):
        return self._decode_strategy_config

    def create_beam(self, requests: list[LlmInferenceExecRequest]) -> BeamGroup:
        beam_group_id = str(uuid4())
        for req in requests:
            req.beam_group_id = beam_group_id

        beam_group = BeamGroup(
            beam_group_id,
            self.decode_strategy_config.n_beams,
            self.decode_strategy_config.temperature,
            requests,
        )
        BeamSearchDecodeStrategy.beam_map[beam_group_id] = beam_group
        return beam_group

    def delete_beam(self, beam_group_id: str):
        beam_group = BeamSearchDecodeStrategy.beam_map[beam_group_id]
        beam_group.clean_up()

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ) -> List[int] | List[List[int]]:
        config = self.decode_strategy_config
        return_metrics = exec_req.llm_inference_metrics is not None
        decode_reqs = [exec_req]
        for _ in range(config.n_beams - 1):
            decode_req = exec_req.replicate_self()
            decode_reqs.append(decode_req)

        beam_group = self.create_beam(decode_reqs)
        for _ in range(config.max_completion_tokens):
            if len(beam_group.completed_reqs) == config.n_beams:
                break

            for exec in beam_group.exec_reqs:
                if exec in beam_group.completed_reqs:
                    continue
                exec.reset(InferencePhase.DECODE)
                config.batcher_callback(exec)

            await beam_group.wait()
            beam_group.process_beams(config.eos_token_id)

        result = []
        metrics = []
        if config.return_top_k:
            reqs = beam_group.completed_reqs
            for req in beam_group.exec_reqs:
                req.llm_inference_metrics.set_end_time()
                reqs.add(req)
            for req in reqs:
                result.append(req.output_token_ids)
                if return_metrics:
                    metrics.append(req.llm_inference_metrics)

        else:
            top_beam = beam_group.find_top_beam()
            result = top_beam.output_token_ids
            if return_metrics:
                if top_beam.llm_inference_metrics.end_time is None:
                    top_beam.llm_inference_metrics.set_end_time()
                metrics.append(top_beam.llm_inference_metrics)

        self.delete_beam(beam_group.beam_group_id)
        if metrics:
            config.streaming_callback(result, metrics)
        else:
            config.streaming_callback(result)
