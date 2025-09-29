import functools
import logging
import time
import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Optional, Generator, AsyncGenerator, Callable, Union

# Configure logger
logger = logging.getLogger("shortfin-llm.tracing")

# Base class for tracing backends
class BaseTracingBackend:
    def init(self, app_name: str) -> None:
        pass

    def frame_enter(self, frame_name: str, request_id: str) -> None:
        pass

    def frame_exit(self, frame_name: str, request_id: str) -> None:
        pass


# Logging-based backend
class LoggerTracingBackend(BaseTracingBackend):
    def __init__(self):
        # Frame tracking - maps (frame_name, request_id) to start time
        self._frames = {}

    def init(self, app_name: str) -> None:
        pass

    def frame_enter(self, frame_name: str, request_id: str) -> None:
        key = (frame_name, request_id)
        self._frames[key] = time.time()

    def frame_exit(self, frame_name: str, request_id: str) -> None:
        key = (frame_name, request_id)
        if key not in self._frames:
            logger.warning(
                f"TRACE: Exit without matching enter for {frame_name} [task={request_id}]"
            )
            return

        duration_ms = round((time.time() - self._frames[key]) * 1e3)
        del self._frames[key]

        msg = f"TRACE: {frame_name} [task={request_id}] completed in {duration_ms}ms"
        logger.info(msg)


# Global tracing configuration
class TracingConfig:
    enabled: bool = True
    app_name: str = "ShortfinLLM"
    backend: BaseTracingBackend = LoggerTracingBackend()
    _initialized: bool = False

    @classmethod
    def is_enabled(cls) -> bool:
        return cls.enabled

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        cls.enabled = enabled
        cls._ensure_initialized()

    @classmethod
    def set_backend(cls, backend_name: str) -> None:
        if backend_name == "log":
            cls.backend = LoggerTracingBackend()
        elif backend_name == "tracy":
            # Import the Tracy backend when requested
            try:
                from .tracy_tracing import TracyTracingBackend

                cls.backend = TracyTracingBackend()
            except ImportError as e:
                logger.error(f"Failed to import Tracy backend: {e}")
                raise ValueError(f"Tracy backend not available: {e}")
        else:
            raise ValueError(f"Unsupported tracing backend: {backend_name}")
        cls._ensure_initialized()

    @classmethod
    def set_app_name(cls, app_name: str) -> None:
        cls.app_name = app_name
        cls._ensure_initialized()

    @classmethod
    def _ensure_initialized(cls) -> None:
        if not cls._initialized and cls.enabled:
            cls.backend.init(cls.app_name)
            cls._initialized = True


# Context managers for manual tracing
@contextmanager
def trace_context(frame_name: str, request_id: str) -> Generator[None, None, None]:
    """Context manager for manual tracing of code blocks."""
    if not TracingConfig.is_enabled():
        yield
        return

    TracingConfig._ensure_initialized()
    TracingConfig.backend.frame_enter(frame_name, request_id)
    try:
        yield
    finally:
        TracingConfig.backend.frame_exit(frame_name, request_id)


@asynccontextmanager
async def async_trace_context(
    frame_name: str, request_id: str
) -> AsyncGenerator[None, None]:
    """Async context manager for manual tracing of code blocks."""
    if not TracingConfig.is_enabled():
        yield
        return

    TracingConfig._ensure_initialized()
    TracingConfig.backend.frame_enter(frame_name, request_id)
    try:
        yield
    finally:
        TracingConfig.backend.frame_exit(frame_name, request_id)
