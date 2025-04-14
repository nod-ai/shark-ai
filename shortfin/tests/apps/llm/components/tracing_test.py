# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import time
import asyncio
import logging
from unittest.mock import patch, MagicMock

from shortfin_apps.llm.components.tracing import (
    LoggerTracingBackend,
    TracingConfig,
    trace_context,
    async_trace_context,
    trace_fn,
)


class TestLoggerTracingBackend:
    def test_frame_tracking(self):
        backend = LoggerTracingBackend()
        backend.init("TestApp")

        backend.frame_enter("test_frame", "task123")
        assert len(backend._frames) == 1
        assert ("test_frame", "task123") in backend._frames

        with patch("logging.Logger.info") as mock_info:
            backend.frame_exit("test_frame", "task123")
            mock_info.assert_called_once()
            assert (
                "TRACE: test_frame [task=task123] completed in"
                in mock_info.call_args[0][0]
            )

        assert len(backend._frames) == 0

    def test_frame_exit_without_enter(self):
        backend = LoggerTracingBackend()
        backend.init("TestApp")

        with patch("logging.Logger.warning") as mock_warning:
            backend.frame_exit("unknown_frame", "task123")
            mock_warning.assert_called_once()
            assert (
                "TRACE: Exit without matching enter for unknown_frame"
                in mock_warning.call_args[0][0]
            )


class TestTraceContext:
    def setup_method(self):
        TracingConfig.enabled = True
        TracingConfig.app_name = "TestApp"
        TracingConfig.backend = LoggerTracingBackend()
        TracingConfig._initialized = False

    def test_sync_trace_context(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend
        op_name = "test_operation"
        request_id = "task123"

        with trace_context(op_name, request_id):
            backend.frame_enter.assert_called_once_with(op_name, request_id)
            backend.frame_exit.assert_not_called()

        backend.frame_exit.assert_called_once_with(op_name, request_id)

    @pytest.mark.asyncio
    async def test_async_trace_context(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend
        op_name = "async_operation"
        request_id = "task456"

        async with async_trace_context(op_name, request_id):
            backend.frame_enter.assert_called_once_with(op_name, request_id)
            backend.frame_exit.assert_not_called()

        backend.frame_exit.assert_called_once_with(op_name, request_id)

    def test_sync_trace_context_with_exception(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend
        op_name = "test_operation"
        request_id = "task123"

        try:
            with trace_context(op_name, request_id):
                raise ValueError("Test exception")
        except ValueError:
            pass

        backend.frame_enter.assert_called_once_with(op_name, request_id)
        backend.frame_exit.assert_called_once_with(op_name, request_id)

    @pytest.mark.asyncio
    async def test_async_trace_context_with_exception(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend
        op_name = "async_operation"
        request_id = "task456"

        try:
            async with async_trace_context(op_name, request_id):
                raise ValueError("Test exception")
        except ValueError:
            pass

        backend.frame_enter.assert_called_once_with(op_name, request_id)
        backend.frame_exit.assert_called_once_with(op_name, request_id)

    def test_trace_context_disabled(self):
        TracingConfig.set_enabled(False)
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        with trace_context("test_operation", "task123"):
            pass

        backend.frame_enter.assert_not_called()
        backend.frame_exit.assert_not_called()


class TestTraceFn:
    def setup_method(self):
        TracingConfig.enabled = True
        TracingConfig.app_name = "TestApp"
        TracingConfig.backend = LoggerTracingBackend()
        TracingConfig._initialized = False

    def test_sync_function_tracing_with_custom_name(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend
        frame_name = "custom_operation"
        expected_result = "result"

        @trace_fn(frame_name)
        def test_function():
            return expected_result

        result = test_function()

        assert result == expected_result
        backend.frame_enter.assert_called_once_with(frame_name, "unknown")
        backend.frame_exit.assert_called_once_with(frame_name, "unknown")

    def test_sync_function_tracing_with_default_name(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend
        expected_result = "result"

        @trace_fn()
        def test_function():
            return expected_result

        result = test_function()

        assert result == expected_result
        backend.frame_enter.assert_called_once_with("test_function", "unknown")
        backend.frame_exit.assert_called_once_with("test_function", "unknown")

    @pytest.mark.asyncio
    async def test_async_function_tracing(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend
        frame_name = "async_operation"
        expected_result = "async result"

        @trace_fn(frame_name)
        async def test_function():
            await asyncio.sleep(0.01)
            return expected_result

        result = await test_function()

        assert result == expected_result
        backend.frame_enter.assert_called_once_with(frame_name, "unknown")
        backend.frame_exit.assert_called_once_with(frame_name, "unknown")

    def test_request_id_extraction(self):
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        class DummyRequest:
            def __init__(self, request_id):
                self.request_id = request_id

        @trace_fn()
        def function_with_request(request):
            return f"Processing {request.request_id}"

        result = function_with_request(DummyRequest("req789"))

        assert result == "Processing req789"
        backend.frame_enter.assert_called_once_with("function_with_request", "req789")
        backend.frame_exit.assert_called_once_with("function_with_request", "req789")
