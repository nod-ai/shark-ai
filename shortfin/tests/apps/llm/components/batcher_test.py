import asyncio
from uuid import uuid4
import pytest

import shortfin.array as sfnp

from unittest.mock import AsyncMock, MagicMock, patch

from shortfin import ProgramIsolation

from shortfin_apps.llm.components.batcher import (
    PrefillExecutorProcess,
    DecodeExecutorProcess,
    LlmBatcherProcess,
    LlmExecutorProcess,
)

from shortfin_apps.llm.components.config_struct import ModelParams, PagedKVCacheParams
from shortfin_apps.llm.components.device_array_cache import (
    Allocation as DeviceArrayAllocation,
    WrappedAllocation as DeviceArrayWrappedAllocation,
)
from shortfin_apps.llm.components.messages import (
    LlmInferenceExecRequest,
    InferencePhase,
)


@pytest.fixture
def model_params():
    return ModelParams(
        max_seq_len=42,
        transformer_block_count=42,
        attn_head_dim=42,
        prefill_batch_sizes=[4],
        decode_batch_sizes=[4],
        paged_kv_cache=PagedKVCacheParams(
            block_seq_stride=42,
            attention_head_count_kv=42,
            device_block_count=256,
            kv_cache_dtype=sfnp.float16,
        ),
    )


@pytest.fixture
def llm_batcher_process(model_params, fiber, cache):
    return LlmBatcherProcess(
        name="test-batcher",
        fiber=fiber,
        page_cache=cache,
        model_params=model_params,
        functions=None,
        ideal_batch_size=4,
        program_isolation=ProgramIsolation.PER_CALL.value,
    )


@pytest.fixture
def llm_executor_process(model_params, fiber, device_array_cache):
    return LlmExecutorProcess(
        name="test-executor",
        fiber=fiber,
        cache=device_array_cache,
        functions=None,
        seq_stride=42,
        page_tables=None,
        program_isolation=ProgramIsolation.PER_CALL.value,
    )


@pytest.fixture
def prefill_executor_process(model_params, fiber, device_array_cache):
    return PrefillExecutorProcess(
        fiber=fiber,
        cache=device_array_cache,
        functions=None,
        seq_stride=model_params.paged_kv_cache.block_seq_stride,
        page_tables=None,
        program_isolation=ProgramIsolation.PER_CALL.value,
    )


@pytest.fixture(scope="function")
def decode_executor_process(model_params, fiber, device_array_cache):
    return DecodeExecutorProcess(
        fiber=fiber,
        cache=device_array_cache,
        functions=None,
        seq_stride=model_params.paged_kv_cache.block_seq_stride,
        page_tables=None,
        program_isolation=ProgramIsolation.PER_CALL.value,
    )


class MockVoidFuture:
    def __init__(self):
        self._event = asyncio.Event()

    def set_success(self):
        self._event.set()

    def __await__(self):
        return self._event.wait().__await__()


@pytest.fixture()
def exec_req_list():
    with patch(
        "shortfin_apps.llm.components.messages.sf.VoidFuture", new=MockVoidFuture
    ):
        input_tokens = [0, 1, 2, 3, 4, 5]

        exec_reqs = []
        for _ in range(4):
            exec_req = LlmInferenceExecRequest(
                phase=InferencePhase.PREFILL,
                input_token_ids=input_tokens,
                rid=str(uuid4()),
                status_tracker=None,
            )
            exec_reqs.append(exec_req)
            input_tokens = [val + 1 for val in input_tokens]

        yield exec_reqs


class TestLlmBatcherProcess:
    @pytest.mark.asyncio
    async def test_board_flights(
        self, llm_batcher_process: LlmBatcherProcess, exec_req_list
    ):
        llm_batcher_process.board = MagicMock()

        ## Empty
        llm_batcher_process.pending = set()
        await llm_batcher_process.board_flights()
        assert llm_batcher_process.board.call_count == 0
        llm_batcher_process.board.reset_mock()

        assert llm_batcher_process.pending == set()

        ## Non-empty
        to_schedule = set(exec_req_list)
        llm_batcher_process.pending = to_schedule
        await llm_batcher_process.board_flights()

        assert llm_batcher_process.board.call_count == 1
        call_args = llm_batcher_process.board.call_args.args
        assert len(call_args) == 3
        assert call_args[0] == llm_batcher_process.page_cache
        assert call_args[1] == llm_batcher_process.fiber
        assert set(call_args[2]) == set(exec_req_list)

        assert llm_batcher_process.pending == set()

    @patch("shortfin_apps.llm.components.batcher.LlmExecutorProcess")
    def test_board(self, mock_executor_cls, llm_batcher_process: LlmBatcherProcess):
        """Test that the board method correctly schedules requests.

        Args:
            mock_executor_cls (MagicMock): Mocked LlmExecutorProcess class.
            llm_batcher_process (LlmBatcherProcess): Instance of LlmBatcherProcess for testing.
        """
        to_schedule = {1, 2, 3, 4}

        mock_executor = MagicMock()
        mock_executor.exec_requests = []
        mock_executor.launch = MagicMock()
        mock_executor_cls.return_value = mock_executor

        llm_batcher_process.make_process = MagicMock(return_value=mock_executor)
        llm_batcher_process.board_request = MagicMock(side_effect=lambda _, x: x)

        llm_batcher_process.board(
            llm_batcher_process.cache,
            llm_batcher_process.fiber,
            to_schedule,
        )

        assert llm_batcher_process.board_request.call_count == len(to_schedule)
        assert mock_executor.launch.call_count == 1


class DummyDeviceArrayAllocation:
    def __init__(self, device_array: sfnp.device_array):
        self.device = device_array
        self.shape = device_array.shape
        self.released = False

    def release(self):
        self.released = True


class TestLlmExecutorProcess:
    def test__transfer_buffer_none_indices(
        self, lsys, llm_executor_process: LlmExecutorProcess, exec_req_list
    ):
        async def test_none_indices():
            llm_executor_process.exec_requests = exec_req_list
            device0 = llm_executor_process.fiber.device(0)

            data = [1, 2, 3, 4, 5, 6]

            buffer0 = sfnp.device_array(device0, [1, 2, 3], dtype=sfnp.float16)
            buffer1 = None

            with buffer0.map(discard=True) as m:
                m.items = data

            logits, indices = await llm_executor_process._transfer_buffer(
                1, device0, [buffer0, buffer1]
            )

            assert isinstance(logits, sfnp.device_array)
            assert indices is None
            assert logits.items.tolist() == [1, 2, 3, 4, 5, 6]

        lsys.run(test_none_indices())

    def test__transfer_buffer_with_indices(
        self, lsys, llm_executor_process: LlmExecutorProcess, exec_req_list
    ):
        async def test_with_indices():
            llm_executor_process.exec_requests = exec_req_list
            device0 = llm_executor_process.fiber.device(0)

            buffer0 = sfnp.device_array(device0, [1, 2, 3], dtype=sfnp.float16)
            buffer1 = sfnp.device_array(device0, [1, 2, 3], dtype=sfnp.int32)

            with buffer0.map(discard=True) as m:
                m.items = [1, 2, 3, 4, 5, 6]

            with buffer1.map(discard=True) as m:
                m.items = [4, 5, 6, 7, 8, 9]

            logits, indices = await llm_executor_process._transfer_buffer(
                1, device0, [buffer0, buffer1]
            )

            assert isinstance(logits, sfnp.device_array)
            assert isinstance(indices, sfnp.device_array)
            assert logits.items.tolist() == [1, 2, 3, 4, 5, 6]
            assert indices.items.tolist() == [4, 5, 6, 7, 8, 9]

        lsys.run(test_with_indices())

    def test_run(self, lsys, llm_executor_process: LlmExecutorProcess, exec_req_list):
        async def test_run_none_indices(*args, **kwargs):
            device0 = llm_executor_process.fiber.device(0)

            async def _dummy_invocation():
                logits = sfnp.device_array(device0, [1, 2, 3], dtype=sfnp.float16)
                indices = None

                with logits.map(discard=True) as m:
                    m.items = [1, 2, 3]
                return logits, indices

            dummy_invocation = AsyncMock(side_effect=_dummy_invocation)

            # Test will fail if it selects the wrong entrypoint
            entrypoints = {
                1: None,
                2: None,
                4: dummy_invocation,
                8: None,
            }

            dummy_arg_input = DummyDeviceArrayAllocation(
                sfnp.device_array(device0, [1, 2, 3], dtype=sfnp.float16)
            )

            llm_executor_process._post_run = AsyncMock()
            llm_executor_process.get_args = AsyncMock(
                return_value=([dummy_arg_input], None)
            )
            llm_executor_process.functions = entrypoints
            llm_executor_process.exec_requests = exec_req_list

            await llm_executor_process.run()
            dummy_invocation.assert_called_once()

        lsys.run(test_run_none_indices())

    def test__post_run_none_indices(
        self, lsys, llm_executor_process: LlmExecutorProcess, exec_req_list
    ):
        async def test_post_run():
            device0 = llm_executor_process.fiber.device(0)
            logits = sfnp.device_array(device0, [1, 2, 3], dtype=sfnp.float16)
            indices = None

            with logits.map(discard=True) as m:
                m.items = [1, 2, 3]

            dummy_arg_input = DummyDeviceArrayAllocation(logits)

            with patch.object(
                LlmInferenceExecRequest, "publish_allocated_pages"
            ) as mock_publish:
                transfered_buffers = []

                llm_executor_process.get_results = AsyncMock(
                    side_effect=lambda logits, indices, count: transfered_buffers.append(
                        (logits, indices)
                    )
                )
                llm_executor_process.exec_requests = exec_req_list

                await llm_executor_process._post_run(
                    [dummy_arg_input], len(exec_req_list), (logits, indices)
                )

                llm_executor_process.get_results.assert_called_once()
                host_logits = transfered_buffers[0][0]
                host_indices = transfered_buffers[0][1]

                assert host_logits.items.tolist() == logits.items.tolist()
                assert host_indices is None

                assert mock_publish.call_count == len(exec_req_list)

        lsys.run(test_post_run())

    def test__post_run_with_indices(
        self, lsys, llm_executor_process: LlmExecutorProcess, exec_req_list
    ):
        async def test_post_run_with_indices():
            device0 = llm_executor_process.fiber.device(0)
            logits = sfnp.device_array(device0, [1, 2, 3], dtype=sfnp.float16)
            indices = sfnp.device_array(device0, [4, 5, 6], dtype=sfnp.int32)

            with logits.map(discard=True) as m:
                m.items = [1, 2, 3]

            with indices.map(discard=True) as m:
                m.items = [4, 5, 6]

            dummy_arg_input = DummyDeviceArrayAllocation(logits)

            with patch.object(
                LlmInferenceExecRequest, "publish_allocated_pages"
            ) as mock_publish:
                transfered_buffers = []

                llm_executor_process.get_results = AsyncMock(
                    side_effect=lambda logits, indices, count: transfered_buffers.append(
                        (logits, indices)
                    )
                )
                llm_executor_process.exec_requests = exec_req_list

                await llm_executor_process._post_run(
                    [dummy_arg_input], len(exec_req_list), (logits, indices)
                )

                llm_executor_process.get_results.assert_called_once()
                host_logits = transfered_buffers[0][0]
                host_indices = transfered_buffers[0][1]

                assert host_logits.items.tolist() == logits.items.tolist()
                assert host_indices.items.tolist() == indices.items.tolist()

                assert mock_publish.call_count == len(exec_req_list)

        lsys.run(test_post_run_with_indices())


class TestPrefillExecutorProcess:
    def test_get_args(
        self, lsys, prefill_executor_process: PrefillExecutorProcess, exec_req_list
    ):
        async def _test_get_args():
            prefill_executor_process.exec_requests = exec_req_list

            page_tables = [
                sfnp.device_array(
                    prefill_executor_process.fiber.device(0),
                    [1, 256],
                    dtype=sfnp.float16,
                )
            ]
            data = [i for i in range(256)]
            with page_tables[0].map(discard=True) as m:
                m.items = data

            prefill_executor_process.page_tables = page_tables

            args, req_count = await prefill_executor_process.get_args(
                len(exec_req_list)
            )
            await prefill_executor_process.fiber.device(0)

            assert len(args) == 4
            assert req_count == len(exec_req_list)

            tokens, seq_lens, seq_block_ids, page_table = args

            for val in [tokens, seq_lens, seq_block_ids]:
                assert isinstance(val, DeviceArrayAllocation)

            assert isinstance(page_table, DeviceArrayWrappedAllocation)
            disable_barrier = page_table.device
            assert isinstance(disable_barrier, sfnp.disable_barrier)

            page_table = disable_barrier.delegate()

            token_device_array = tokens.device
            for i in range(len(exec_req_list)):
                input_tokens = token_device_array.view(i).items.tolist()
                expected_tokens = exec_req_list[i].input_token_ids
                assert input_tokens[0 : len(expected_tokens)] == expected_tokens

            seq_lens = seq_lens.device.items.tolist()
            seq_block_ids = seq_block_ids.device.items.tolist()
            page_table = page_table.items.tolist()

            assert seq_lens == [6, 6, 6, 6]
            assert seq_block_ids == [0, 0, 0, 0]
            assert page_table == data

        lsys.run(_test_get_args())

    def test_get_results_none_indices(
        self,
        lsys,
        prefill_executor_process: PrefillExecutorProcess,
        exec_req_list: list[LlmInferenceExecRequest],
    ):
        async def _test_get_results():
            sl = len(exec_req_list[0].input_token_ids)

            logits = sfnp.device_array(
                prefill_executor_process.fiber.device(0),
                [4, sl, 16],
                dtype=sfnp.float16,
            )

            data = [i for i in range(16)]
            for i in range(len(exec_req_list)):
                with logits.view(i, sl - 1).map(discard=True) as m:
                    m.items = data

            indices = None
            prefill_executor_process.exec_requests = exec_req_list
            await prefill_executor_process.get_results(
                logits, indices, len(exec_req_list)
            )
            await prefill_executor_process.fiber.device(0)

            for req in exec_req_list:
                logits = req.result_logits.items.tolist()
                assert logits == [i for i in range(16)]

                assert req.result_indices is None
                assert req.done

        lsys.run(_test_get_results())

    def test_get_results_with_indices(
        self,
        lsys,
        prefill_executor_process: PrefillExecutorProcess,
        exec_req_list: list[LlmInferenceExecRequest],
    ):
        async def _test_get_results():
            sl = len(exec_req_list[0].input_token_ids)

            logits = sfnp.device_array(
                prefill_executor_process.fiber.device(0),
                [4, sl, 16],
                dtype=sfnp.float16,
            )
            indices = sfnp.device_array(
                prefill_executor_process.fiber.device(0),
                [4, sl, 16],
                dtype=sfnp.int64,
            )

            data = [i for i in range(16)]
            for i in range(len(exec_req_list)):
                with logits.view(i, sl - 1).map(discard=True) as m:
                    m.items = data
                with indices.view(i, sl - 1).map(discard=True) as m:
                    m.items = data

            prefill_executor_process.exec_requests = exec_req_list
            await prefill_executor_process.get_results(
                logits, indices, len(exec_req_list)
            )
            await prefill_executor_process.fiber.device(0)

            for req in exec_req_list:
                logits = req.result_logits.items.tolist()
                indices = req.result_indices.items.tolist()

                assert logits == [i for i in range(16)]
                assert indices == [i for i in range(16)]

                assert req.done

        lsys.run(_test_get_results())


class TestDecodeExecutorProcess:
    def test_get_args(
        self,
        lsys,
        decode_executor_process: DecodeExecutorProcess,
        exec_req_list: list[LlmInferenceExecRequest],
    ):
        async def _test_get_args():
            for req in exec_req_list:
                req.start_position = len(req.input_token_ids) - 1

            decode_executor_process.exec_requests = exec_req_list

            page_tables = [
                sfnp.device_array(
                    decode_executor_process.fiber.device(0),
                    [1, 256],
                    dtype=sfnp.float16,
                )
            ]
            data = [i for i in range(256)]
            with page_tables[0].map(discard=True) as m:
                m.items = data

            decode_executor_process.page_tables = page_tables

            args, req_count = await decode_executor_process.get_args(len(exec_req_list))
            await decode_executor_process.fiber.device(0)

            assert len(args) == 5
            assert req_count == len(exec_req_list)

            tokens, start_positions, seq_lens, seq_block_ids, page_table = args

            for val in [tokens, start_positions, seq_lens, seq_block_ids]:
                assert isinstance(val, DeviceArrayAllocation)

            assert isinstance(page_table, DeviceArrayWrappedAllocation)
            disable_barrier = page_table.device
            assert isinstance(disable_barrier, sfnp.disable_barrier)

            page_table = disable_barrier.delegate()

            token_device_array = tokens.device
            for i in range(len(exec_req_list)):
                input_tokens = token_device_array.view(i).items.tolist()
                expected_tokens = exec_req_list[i].input_token_ids[-1]
                assert input_tokens == [expected_tokens]

            start_positions = start_positions.device.items.tolist()
            seq_lens = seq_lens.device.items.tolist()
            seq_block_ids = seq_block_ids.device.items.tolist()
            page_table = page_table.items.tolist()

            assert start_positions == [6, 6, 6, 6]
            assert seq_lens == [5, 5, 5, 5]
            assert seq_block_ids == [0, 0, 0, 0]
            assert page_table == data

        lsys.run(_test_get_args())

    def test_get_results_none_indices(
        self,
        lsys,
        decode_executor_process: DecodeExecutorProcess,
        exec_req_list: list[LlmInferenceExecRequest],
    ):
        async def _test_get_results():
            sl = len(exec_req_list[0].input_token_ids)

            logits_input = sfnp.device_array(
                decode_executor_process.fiber.device(0),
                [4, sl, 16],
                dtype=sfnp.float16,
            )

            data = [i for i in range(16)]
            for i in range(len(exec_req_list)):
                with logits_input.view(i, 0).map(discard=True) as m:
                    m.items = data

            indices = None
            decode_executor_process.exec_requests = exec_req_list
            await decode_executor_process.get_results(
                logits_input, indices, len(exec_req_list)
            )
            await decode_executor_process.fiber.device(0)

            for req in exec_req_list:
                logits = req.result_logits.items.tolist()
                assert logits == data

                assert req.result_indices is None
                assert req.done._event.is_set()

        lsys.run(_test_get_results())

    def test_get_results_with_indices(
        self,
        lsys,
        decode_executor_process: DecodeExecutorProcess,
        exec_req_list: list[LlmInferenceExecRequest],
    ):
        async def _test_get_results():
            sl = 1

            logits = sfnp.device_array(
                decode_executor_process.fiber.device(0),
                [4, sl, 16],
                dtype=sfnp.float16,
            )
            indices = sfnp.device_array(
                decode_executor_process.fiber.device(0),
                [4, sl, 16],
                dtype=sfnp.int64,
            )

            data = [i for i in range(16)]
            for i in range(len(exec_req_list)):
                with logits.view(i, 0).map(discard=True) as m:
                    m.items = data
                with indices.view(i, 0).map(discard=True) as m:
                    m.items = data

            decode_executor_process.exec_requests = exec_req_list
            await decode_executor_process.get_results(
                logits, indices, len(exec_req_list)
            )
            await decode_executor_process.fiber.device(0)

            for req in exec_req_list:
                logits = req.result_logits.items.tolist()
                indices = req.result_indices.items.tolist()

                assert logits == data
                assert indices == data

                assert req.done

        lsys.run(_test_get_results())
