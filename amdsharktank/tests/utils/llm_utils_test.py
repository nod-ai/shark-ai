from unittest import TestCase
from unittest.mock import MagicMock, patch

from amdsharktank.utils.llm_scheduler import BasicScheduler, ChunkScheduler
from amdsharktank.utils.llm_tasks import LlmRequest, LlmTaskInput
from amdsharktank.utils.llm_utils import (
    make_chunks,
    IreeInstance,
    LlmRunner,
)


class TestMakeChunks(TestCase):
    def setUp(self):
        self._block_seq_stride = 2

    def test_single_chunk(self):
        llm_request = LlmRequest(
            request_id="req-0",
            tokens=[1, 2],
            pages=[0],
        )

        expected = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=0,
                tokens=[1, 2],
                seq_len=2,
                pages=[0],
                start_position=0,
            )
        ]

        actual = make_chunks(
            llm_request, chunk_block_size=1, block_stride=self._block_seq_stride
        )

        self.assertEqual(expected, actual)

    def test_multiple_chunks_all_full(self):
        # Full chunks
        llm_request = LlmRequest(
            request_id="req-0",
            tokens=[1, 2, 3, 4, 5, 6],
            pages=[0, 1, 2],
        )

        expected = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=0,
                tokens=[1, 2],
                seq_len=2,
                pages=[0],
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-0",
                chunk_id=1,
                tokens=[3, 4],
                seq_len=4,
                pages=[0, 1],
                start_position=2,
            ),
            LlmTaskInput(
                request_id="req-0",
                chunk_id=2,
                tokens=[5, 6],
                seq_len=6,
                pages=[0, 1, 2],
                start_position=4,
            ),
        ]

        actual = make_chunks(
            llm_request, chunk_block_size=1, block_stride=self._block_seq_stride
        )
        self.assertEqual(expected, actual)

    def test_multiple_chunks_last_partial(self):
        # Test to validate we handle the case where
        # len(input_tokens) is not divisible by chunk_size * block_stride.
        # This makes the last chunk a `partial` chunk.
        llm_request = LlmRequest(
            request_id="req-0",
            tokens=[1, 2, 3, 4, 5],
            pages=[0, 1, 2],
        )

        expected = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=0,
                tokens=[1, 2],
                seq_len=2,
                pages=[0],
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-0",
                chunk_id=1,
                tokens=[3, 4],
                seq_len=4,
                pages=[0, 1],
                start_position=2,
            ),
            LlmTaskInput(
                request_id="req-0",
                chunk_id=2,
                tokens=[5],
                seq_len=5,
                pages=[0, 1, 2],
                start_position=4,
            ),
        ]

        actual = make_chunks(
            llm_request, chunk_block_size=1, block_stride=self._block_seq_stride
        )
        self.assertEqual(expected, actual)


class TestLlmRunner(TestCase):
    def setUp(self):
        self._prefill_bs = 4
        self._decode_bs = 8
        self._mock_iree_instance = MagicMock(spec=IreeInstance)
        self._mock_iree_instance.prefill_bs = self._prefill_bs
        self._mock_iree_instance.decode_bs = self._decode_bs
        self._page_count = 16
        self._page_sizes = [256]
        self._block_stride = 2
        self._kv_cache_dtype = "float16"

        self._llm_runner = LlmRunner(
            instance=self._mock_iree_instance,
            page_count=self._page_count,
            page_sizes=self._page_sizes,
            block_stride=self._block_stride,
            kv_cache_dtype=self._kv_cache_dtype,
        )

    def test_make_requests(self):
        requests = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ]
        page_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
            [10, 11, 12, 13],
        ]

        llm_runner = self._llm_runner
        llm_requests = llm_runner.make_requests(requests=requests, page_ids=page_ids)

        expected_requests = [
            LlmRequest(
                request_id="req-0",
                tokens=[0, 1, 2, 3, 4, 5, 6],
                pages=[0, 1, 2, 3],
            ),
            LlmRequest(
                request_id="req-1",
                tokens=[1, 2, 3, 4, 5, 6, 7],
                pages=[4, 5, 6, 7],
            ),
            LlmRequest(
                request_id="req-2",
                tokens=[2, 3, 4, 5],
                pages=[8, 9],
            ),
            LlmRequest(
                request_id="req-3",
                tokens=[3, 4, 5, 6, 7, 8, 9, 10],
                pages=[10, 11, 12, 13],
            ),
        ]

        self.assertEqual(expected_requests, llm_requests)

    def test_submit_prefill_no_chunking(self):
        requests = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ]
        page_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
            [10, 11, 12, 13],
        ]

        submitted_task_inputs = []
        llm_runner = self._llm_runner
        llm_requests = llm_runner.make_requests(requests=requests, page_ids=page_ids)
        with patch.object(
            BasicScheduler,
            "schedule_task",
            side_effect=lambda task: submitted_task_inputs.append(task),
        ):
            llm_runner.submit_prefill(llm_requests)

        expected_task_inputs = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=0,
                tokens=[0, 1, 2, 3, 4, 5, 6],
                seq_len=7,
                pages=[0, 1, 2, 3],
                start_position=None,
            ),
            LlmTaskInput(
                request_id="req-1",
                chunk_id=0,
                tokens=[1, 2, 3, 4, 5, 6, 7],
                seq_len=7,
                pages=[4, 5, 6, 7],
                start_position=None,
            ),
            LlmTaskInput(
                request_id="req-2",
                chunk_id=0,
                tokens=[2, 3, 4, 5],
                seq_len=4,
                pages=[8, 9],
                start_position=None,
            ),
            LlmTaskInput(
                request_id="req-3",
                chunk_id=0,
                tokens=[3, 4, 5, 6, 7, 8, 9, 10],
                seq_len=8,
                pages=[10, 11, 12, 13],
                start_position=None,
            ),
        ]

        self.assertEqual(expected_task_inputs, submitted_task_inputs)

    def test_submit_decode(self):
        requests = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ]
        page_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
            [10, 11, 12, 13],
        ]

        submitted_task_inputs = []
        llm_runner = self._llm_runner
        llm_requests = llm_runner.make_requests(requests=requests, page_ids=page_ids)
        with patch.object(
            BasicScheduler,
            "schedule_task",
            side_effect=lambda task: submitted_task_inputs.append(task),
        ):
            llm_runner.submit_decode(llm_requests)

        expected_task_inputs = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=0,
                tokens=[6],
                seq_len=7,
                pages=[0, 1, 2, 3],
                start_position=6,
            ),
            LlmTaskInput(
                request_id="req-1",
                chunk_id=0,
                tokens=[7],
                seq_len=7,
                pages=[4, 5, 6, 7],
                start_position=6,
            ),
            LlmTaskInput(
                request_id="req-2",
                chunk_id=0,
                tokens=[5],
                seq_len=4,
                pages=[8, 9],
                start_position=3,
            ),
            LlmTaskInput(
                request_id="req-3",
                chunk_id=0,
                tokens=[10],
                seq_len=8,
                pages=[10, 11, 12, 13],
                start_position=7,
            ),
        ]

        self.assertEqual(expected_task_inputs, submitted_task_inputs)


class TestLlmRunnerWithChunking(TestCase):
    def setUp(self):
        self._prefill_bs = 4
        self._decode_bs = 8
        self._mock_iree_instance = MagicMock(spec=IreeInstance)
        self._mock_iree_instance.prefill_bs = self._prefill_bs
        self._mock_iree_instance.decode_bs = self._decode_bs
        self._page_count = 16
        self._page_sizes = [256]
        self._block_stride = 2
        self._kv_cache_dtype = "float16"
        self._chunk_block_size = 2

        self._llm_runner = LlmRunner(
            instance=self._mock_iree_instance,
            page_count=self._page_count,
            page_sizes=self._page_sizes,
            block_stride=self._block_stride,
            kv_cache_dtype=self._kv_cache_dtype,
            chunk_block_size=self._chunk_block_size,
        )

    def test_submit_prefill(self):
        requests = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ]
        page_ids = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
            [10, 11, 12, 13],
        ]

        submitted_task_inputs = []
        llm_runner = self._llm_runner
        llm_requests = llm_runner.make_requests(requests=requests, page_ids=page_ids)
        with patch.object(
            ChunkScheduler,
            "schedule_task",
            side_effect=lambda task: submitted_task_inputs.append(task),
        ):
            llm_runner.submit_prefill(llm_requests)

        expected_task_inputs = [
            LlmTaskInput(
                request_id="req-0",
                chunk_id=0,
                tokens=[0, 1, 2, 3],
                seq_len=4,
                pages=[0, 1],
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-0",
                chunk_id=1,
                tokens=[4, 5, 6],
                seq_len=7,
                pages=[0, 1, 2, 3],
                start_position=4,
            ),
            LlmTaskInput(
                request_id="req-1",
                chunk_id=0,
                tokens=[1, 2, 3, 4],
                seq_len=4,
                pages=[4, 5],
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-1",
                chunk_id=1,
                tokens=[5, 6, 7],
                seq_len=7,
                pages=[4, 5, 6, 7],
                start_position=4,
            ),
            LlmTaskInput(
                request_id="req-2",
                chunk_id=0,
                tokens=[2, 3, 4, 5],
                seq_len=4,
                pages=[8, 9],
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-3",
                chunk_id=0,
                tokens=[3, 4, 5, 6],
                seq_len=4,
                pages=[10, 11],
                start_position=0,
            ),
            LlmTaskInput(
                request_id="req-3",
                chunk_id=1,
                tokens=[7, 8, 9, 10],
                seq_len=8,
                pages=[10, 11, 12, 13],
                start_position=4,
            ),
        ]

        self.assertEqual(expected_task_inputs, submitted_task_inputs)
