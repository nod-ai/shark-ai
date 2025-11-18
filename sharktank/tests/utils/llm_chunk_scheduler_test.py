from typing import Any, List
from iree.runtime.array_interop import DeviceArray
import numpy

from unittest import TestCase
from unittest.mock import MagicMock, patch

from torch._tensor import Tensor

from amdsharktank.utils.llm_tasks import LlmTaskInput, LlmTask, PrefillTask
from amdsharktank.utils.llm_scheduler import BasicScheduler, ChunkScheduler
from amdsharktank.utils.llm_utils import LlmRunner, IreeInstance, dtype_string_to_type


class RecordingDummyLlmTask(LlmTask):
    constructed_batches = []

    def __init__(
        self,
        invocation_fn,
        llm_task_inputs: List[LlmTaskInput],
        batch_size: int,
        block_stride: int,
        has_prefill_position: bool = False,
        chunk_block_size: int = None,
    ):
        super().__init__(invocation_fn, llm_task_inputs, batch_size, block_stride)
        self.__class__.constructed_batches.append(llm_task_inputs)

    def _prepare_args(
        self, task_inputs: List[LlmTaskInput], *cache
    ) -> List[numpy.ndarray[tuple[Any, ...], numpy.dtype[Any]] | DeviceArray | Tensor]:
        return [
            numpy.array(
                [len(t.tokens) for t in task_inputs],
                dtype=dtype_string_to_type["int64"],
            )
        ]

    def _process_results(
        self,
        results: List[
            numpy.ndarray[tuple[Any, ...], numpy.dtype[Any]] | DeviceArray | Tensor
        ],
    ):
        return results

    @property
    def logit_positions(self) -> List[int]:
        return [task_input.seq_len - 1 for task_input in self._task_inputs]

    def run(self, _):
        return (
            [i for i in range(len(self._task_inputs))],
            None,
        )


class TestChunkScheduler(TestCase):
    def setUp(self) -> None:
        self._batch_size = 4
        self._mock_iree_instance = MagicMock(spec=IreeInstance)
        self._mock_iree_instance.prefill_bs = self._batch_size
        self._mock_iree_instance.decode_bs = self._batch_size
        self._page_count = 16
        self._page_sizes = [256]
        self._block_stride = 2
        self._kv_cache_dtype = "float16"
        self._cache = [
            numpy.zeros(
                (self._page_count, self._page_sizes[0]),
                dtype=dtype_string_to_type[self._kv_cache_dtype],
            )
        ]
        self._chunk_block_size = 2

        self._llm_runner = LlmRunner(
            instance=self._mock_iree_instance,
            page_count=self._page_count,
            page_sizes=self._page_sizes,
            block_stride=self._block_stride,
            kv_cache_dtype=self._kv_cache_dtype,
            chunk_block_size=self._chunk_block_size,
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

        self._chunk_scheduler = ChunkScheduler(
            batch_size=self._batch_size,
            block_seq_stride=self._block_stride,
            llm_task_class=RecordingDummyLlmTask,
            invocation_fn=self.dummy_invocation_fn,
            has_prefill_position=True,
            chunk_block_size=2,
        )

    def dummy_invocation_fn(self):
        pass

    def dummy_selection_fn(self, logits, indices, logit_positions):
        return logit_positions

    def test_run(self):
        RecordingDummyLlmTask.constructed_batches = []

        chunk_scheduler = self._chunk_scheduler
        llm_runner = self._llm_runner

        requests = self._requests
        page_ids = self._page_ids

        llm_requests = llm_runner.make_requests(requests, page_ids)
        llm_runner._prefill_scheduler = chunk_scheduler

        llm_runner.submit_prefill(llm_requests)
        with patch.object(PrefillTask, "__init__", RecordingDummyLlmTask.__init__):
            with patch.object(PrefillTask, "run", RecordingDummyLlmTask.run):
                selections = chunk_scheduler.run(
                    selection_fn=self.dummy_selection_fn, cache=self._cache
                )
        constructed_batches = RecordingDummyLlmTask.constructed_batches

        expected_batch_0 = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=0,
                tokens=requests[0][:4],
                pages=page_ids[0][:2],
                seq_len=4,
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-1",
                chunk_id=0,
                tokens=requests[1][:4],
                pages=page_ids[1][:2],
                seq_len=4,
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-2",
                chunk_id=0,
                tokens=requests[2],
                pages=page_ids[2],
                seq_len=4,
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-3",
                chunk_id=0,
                tokens=requests[3][:4],
                pages=page_ids[3][:2],
                seq_len=4,
                start_position=0,
            ),
        ]
        expected_batch_1 = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=1,
                tokens=requests[0][4:],
                pages=page_ids[0],
                seq_len=len(requests[0]),
                start_position=4,
            ),
            LlmTaskInput(
                request_id="req-1",
                chunk_id=1,
                tokens=requests[1][4:],
                pages=page_ids[1],
                seq_len=len(requests[1]),
                start_position=4,
            ),
            LlmTaskInput(
                request_id="req-3",
                chunk_id=1,
                tokens=requests[3][4:],
                pages=page_ids[3],
                seq_len=len(requests[3]),
                start_position=4,
            ),
        ]

        self.assertEqual(len(constructed_batches), 2)
        self.assertEqual(constructed_batches[0], expected_batch_0)
        self.assertEqual(constructed_batches[1], expected_batch_1)

        expected_selections = {
            "req-0": [3, 6],
            "req-1": [3, 6],
            "req-2": [3],
            "req-3": [3, 7],
        }
        self.assertEqual(selections, expected_selections)
