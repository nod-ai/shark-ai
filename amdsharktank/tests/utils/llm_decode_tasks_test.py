from typing import List

import numpy

from unittest import TestCase
from unittest.mock import MagicMock, patch

from amdsharktank.utils.llm_scheduler import BasicScheduler, ChunkScheduler
from amdsharktank.utils.llm_tasks import (
    LlmTaskInput,
    DecodeTask,
)

from amdsharktank.utils.llm_utils import (
    dtype_string_to_type,
    IreeInstance,
    LlmRunner,
)


class TestDecodeTask(TestCase):
    def setUp(self) -> None:
        self._batch_size = 4
        self._mock_iree_instance = MagicMock(spec=IreeInstance)
        self._mock_iree_instance.prefill_bs = self._batch_size
        self._mock_iree_instance.decode_bs = self._batch_size
        self._page_count = 16
        self._page_sizes = [256]
        self._block_stride = 2
        self._kv_cache_dtype = "float16"
        self._cache_state = numpy.zeros(
            (self._page_count, self._page_sizes[0]),
            dtype=dtype_string_to_type[self._kv_cache_dtype],
        )

        self._int_dtype = dtype_string_to_type["int64"]

        self._llm_runner = LlmRunner(
            instance=self._mock_iree_instance,
            page_count=self._page_count,
            page_sizes=self._page_sizes,
            block_stride=self._block_stride,
            kv_cache_dtype=self._kv_cache_dtype,
        )

        self._requests = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ]
        self._page_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
            [10, 11, 12, 13],
        ]

    def _get_llm_task_inputs(self, requests, page_ids) -> List[LlmTaskInput]:
        llm_runner = self._llm_runner
        llm_requests = llm_runner.make_requests(requests=requests, page_ids=page_ids)
        llm_task_inputs = []
        with patch.object(
            BasicScheduler,
            "schedule_task",
            side_effect=lambda task: llm_task_inputs.append(task),
        ):
            llm_runner.submit_decode(llm_requests)

        return llm_task_inputs

    def test_run(self):
        requests = self._requests
        page_ids = self._page_ids
        llm_task_inputs = self._get_llm_task_inputs(requests, page_ids)

        invoked_args = []

        def _invocation_fn(*args):
            invoked_args.extend(args)
            return numpy.array([i for i in range(self._batch_size)])

        decode_task = DecodeTask(
            invocation_fn=_invocation_fn,
            llm_task_inputs=llm_task_inputs,
            batch_size=self._batch_size,
            block_stride=self._block_stride,
            decode_topk_logits=None,
        )
        logits, indices = decode_task.run(self._cache_state)

        expected_tokens = numpy.array(
            [[request[-1]] for request in requests], dtype=self._int_dtype
        )
        expected_seq_len = numpy.array(
            [llm_task.seq_len for llm_task in llm_task_inputs], dtype=self._int_dtype
        )
        expected_start_positions = numpy.array(
            [llm_task.start_position for llm_task in llm_task_inputs],
            dtype=self._int_dtype,
        )
        expected_seq_block_ids = numpy.array(
            [
                page_ids[0],
                page_ids[1],
                page_ids[2] + [0, 0],
                page_ids[3],
            ],
            dtype=self._int_dtype,
        )
        assert numpy.array_equal(invoked_args[0], expected_tokens)
        assert numpy.array_equal(invoked_args[1], expected_seq_len)
        assert numpy.array_equal(invoked_args[2], expected_start_positions)
        assert numpy.array_equal(invoked_args[3], expected_seq_block_ids)
        assert numpy.array_equal(invoked_args[4], self._cache_state)

        assert numpy.array_equal(
            logits,
            numpy.array([i for i in range(self._batch_size)], dtype=self._int_dtype),
        )
        assert numpy.array_equal(
            indices,
            numpy.array([i for i in range(self._batch_size)], dtype=self._int_dtype),
        )

    def test_run_w_indices(self):
        requests = self._requests
        page_ids = self._page_ids
        llm_task_inputs = self._get_llm_task_inputs(requests, page_ids)

        invoked_args = []

        def _invocation_fn(*args):
            invoked_args.extend(args)
            return (
                [i for i in range(self._batch_size)],
                [i + 1 for i in range(self._batch_size)],
            )

        decode_task = DecodeTask(
            invocation_fn=_invocation_fn,
            llm_task_inputs=llm_task_inputs,
            batch_size=self._batch_size,
            block_stride=self._block_stride,
            decode_topk_logits=None,
        )

        logits, indices = decode_task.run(self._cache_state)

        assert numpy.array_equal(
            logits,
            numpy.array([i for i in range(self._batch_size)], dtype=self._int_dtype),
        )
        assert numpy.array_equal(
            indices,
            numpy.array(
                [i + 1 for i in range(self._batch_size)], dtype=self._int_dtype
            ),
        )

    def test_run_w_topk(self):
        requests = self._requests
        page_ids = self._page_ids
        llm_task_inputs = self._get_llm_task_inputs(requests, page_ids)

        invoked_args = []

        def _invocation_fn(*args):
            invoked_args.extend(args)
            return [i for i in range(self._batch_size)][::-1]

        decode_task = DecodeTask(
            invocation_fn=_invocation_fn,
            llm_task_inputs=llm_task_inputs,
            batch_size=self._batch_size,
            block_stride=self._block_stride,
            decode_topk_logits=2,
        )

        logits, indices = decode_task.run(self._cache_state)

        assert numpy.array_equal(
            logits,
            numpy.array(
                [3, 2],
                dtype=self._int_dtype,
            ),
        )
        assert numpy.array_equal(
            indices,
            numpy.array(
                [0, 1],
                dtype=self._int_dtype,
            ),
        )
