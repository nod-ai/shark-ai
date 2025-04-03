import asyncio
import logging
import math
import pytest
import random
from typing import Any, List
from unittest.mock import patch

from shortfin_apps.llm.components.kvcache.base_attention_cache import (
    BasePagedAttentionCacheAllocation,
)
from shortfin_apps.llm.components.messages import LlmInferenceExecRequest
from shortfin_apps.llm.components.token_selection_strategy.beam_group import (
    BeamGroup,
    Beam,
)


import shortfin.array as sfnp


@pytest.fixture()
def exec_req_list(exec_req, cache, dummy_pages):
    exec_req._cache = cache
    allocation = BasePagedAttentionCacheAllocation(dummy_pages, cache=cache)
    exec_req.allocation = allocation
    exec_reqs = [exec_req]
    num_beams = len(dummy_pages)
    with patch.object(exec_req._cache, "fork_pages", return_value=allocation):
        for _ in range(num_beams - 1):
            exec_reqs.append(LlmInferenceExecRequest.copy_exec_request(exec_req))

    yield exec_reqs


def approximately_equal(a: Any, b: Any, rel_tol=1e-2, abs_tol=0.0) -> bool:
    """
    Recursively checks if two nested lists (or scalar values) are approximately equal.

    Args:
        a: First list or scalar.
        b: Second list or scalar.
        rel_tol: Relative tolerance.
        abs_tol: Absolute tolerance.

    Returns:
        True if all corresponding elements are approximately equal.
    """
    # If both are lists, iterate element-wise
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(
            approximately_equal(sub_a, sub_b, rel_tol, abs_tol)
            for sub_a, sub_b in zip(a, b)
        )

    # Otherwise, assume they are scalars and compare
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


class DummyBeam(Beam):
    def sample_logits(self):
        pass

    def update_score(self, value: float):
        pass

    def update_exec_req(self):
        pass

    def normalize_score(self, value: float):
        pass

    def update_final_score(self):
        pass


def test_beam_apply_temperature(device, exec_req):
    """Test that `apply_temperature` works correctly on the `result_logits`.

    Args:
        exec_req (LlmInferenceExecRequest): Request to apply `temperature` too.
    """
    value = float(42)
    src = sfnp.device_array(device, [1, 1, 16], dtype=sfnp.float32)
    data = [value for _ in range(math.prod(src.shape))]
    src.items = data
    exec_req.result_logits = src

    temperature = 1.0
    beam = DummyBeam(
        exec_req,
        temperature,
    )

    with patch.object(sfnp, "divide") as temp_mock:
        expected = value / temperature
        beam.apply_temperature()
        logits = beam.exec_req.result_logits.items.tolist()
        assert all(approximately_equal(expected, logit) for logit in logits)
        temp_mock.assert_not_called()

    temperature = 0.5
    beam.temperature = temperature
    expected = value / temperature
    beam.apply_temperature()
    logits = beam.exec_req.result_logits.items.tolist()
    assert all(approximately_equal(expected, logit) for logit in logits)

    temperature = 1.5
    beam.exec_req.result_logits.items = data
    beam.temperature = temperature
    expected = value / temperature
    beam.apply_temperature()
    logits = beam.exec_req.result_logits.items.tolist()
    assert all(approximately_equal(expected, logit) for logit in logits)


@pytest.mark.asyncio
async def test_wait(exec_req_list):
    async def set_done(exec_reqs: List[LlmInferenceExecRequest]):
        for req in exec_reqs:
            req.done.set_success()

    beams = [DummyBeam(exec_req) for exec_req in exec_req_list]
    beam_groups = BeamGroup(
        eos_token_id=-1,
        num_beams=len(exec_req_list),
        beams=beams,
        selection_callback=lambda x: None,
    )
    await asyncio.gather(*[beam_groups.wait(), set_done(exec_req_list)])
    for req in exec_req_list:
        assert req.done._event.is_set()


def test_process_beams_one_req(exec_req):
    def selection_callback(active_beams: List[DummyBeam], _: List[DummyBeam]):
        selections = []
        for beam in active_beams:
            token = 0
            beam.last_token = token
            selections.append(beam)

        return selections

    beams = [DummyBeam(exec_req)]
    beam_groups = BeamGroup(
        eos_token_id=-1,
        num_beams=1,
        beams=beams,
        selection_callback=selection_callback,
    )

    # Active
    beam_groups.process_beams()
    assert beam_groups.active_beams == beams
    assert len(beam_groups.completed_beams) == 0

    # Completed
    beam_groups.eos_token_id = 0
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        beam_groups.process_beams()
        assert len(beam_groups.active_beams) == 0
        assert beam_groups.completed_beams == beams
        free_cache_mock.assert_called_once()


def test_process_beams_multiple_reqs(exec_req_list):
    def selection_callback_no_completed(active_beams, _):
        selections = []
        for beam in active_beams:
            token = 0
            beam.last_token = token
            selections.append(beam)
        return selections

    def selection_callback_one_completed(active_beams, _):
        active_beams[0].last_token = 1
        selections = [active_beams[0]]
        for beam in active_beams[1:]:
            beam.last_token = 0
            selections.append(
                beam,
            )
        return selections

    def selection_callback_all_completed(active_beams, _):
        selections = []
        for beam in active_beams:
            beam.last_token = 1
            selections.append(
                beam,
            )
        return selections

    req_list = exec_req_list.copy()
    beams = [DummyBeam(req) for req in req_list]
    beam_group = BeamGroup(
        eos_token_id=1,
        num_beams=len(req_list),
        beams=beams,
        selection_callback=selection_callback_no_completed,
    )
    beam_group.process_beams()
    assert beam_group.active_beams == beams
    assert len(beam_group.completed_beams) == 0

    req_list = exec_req_list.copy()
    beams = [DummyBeam(req) for req in req_list]
    beam_group = BeamGroup(
        eos_token_id=1,
        num_beams=len(req_list),
        beams=beams,
        selection_callback=selection_callback_one_completed,
    )
    expected = [beam_group.active_beams[0]]
    active = beam_group.active_beams[1:]
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        beam_group.selection_callback = selection_callback_one_completed
        beam_group.process_beams()
        assert beam_group.active_beams == active
        assert beam_group.completed_beams == expected
        free_cache_mock.assert_called_once()

        # Complete another req
        expected.append(beam_group.active_beams[0])
        active.remove(beam_group.active_beams[0])
        beam_group.process_beams()
        assert beam_group.active_beams == active
        assert beam_group.completed_beams == expected
        assert free_cache_mock.call_count == 2

    req_list = exec_req_list.copy()
    beams = [DummyBeam(req) for req in req_list]
    beam_group = BeamGroup(
        eos_token_id=1,
        num_beams=len(req_list),
        beams=beams,
        selection_callback=selection_callback_all_completed,
    )
    # All completed
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        beam_group.process_beams()
        assert len(beam_group.active_beams) == 0
        assert beam_group.completed_beams == beams
        assert free_cache_mock.call_count == len(beams)


@pytest.mark.asyncio
async def test_clean_up(exec_req_list):
    beams = [DummyBeam(req) for req in exec_req_list]
    beam_group = BeamGroup(
        eos_token_id=-1,
        num_beams=len(exec_req_list),
        beams=beams,
        selection_callback=lambda x: None,
    )
    with patch.object(LlmInferenceExecRequest, "free_cache_pages") as free_cache_mock:
        # All active
        beam_group.clean_up()
        assert free_cache_mock.call_count == len(beam_group.active_beams)

        free_cache_mock.reset_mock()

        # All completed
        beam_group.completed_beams = beams
        beam_group.active_beams = []
        beam_group.clean_up()
        assert free_cache_mock.call_count == len(beam_group.completed_beams)

        free_cache_mock.reset_mock()

        # Mixture of both
        beam_group.completed_beams = beams[: len(exec_req_list) // 2]
        beam_group.active_beams = beams[len(exec_req_list) // 2 :]
        beam_group.clean_up()
        assert free_cache_mock.call_count == len(beam_group.completed_beams) + len(
            beam_group.active_beams
        )
