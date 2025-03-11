import functools
import logging
import time
import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Optional, Generator, AsyncGenerator, Callable, Union

# Configure logger
logger = logging.getLogger("shortfin-llm.tracing")

# Base class for tracing backends
class TracingBackend:
    def init(self, app_name: str) -> None:
        pass

    def frame_enter(self, frame_name: str, task_id: str) -> None:
        pass

    def frame_exit(self, frame_name: str, task_id: str) -> None:
        pass


# Logging-based backend
class LoggingBackend(TracingBackend):
    def __init__(self):
        # Frame tracking - maps (frame_name, task_id) to start time
        self._frames = {}

    def init(self, app_name: str) -> None:
        pass

    def frame_enter(self, frame_name: str, task_id: str) -> None:
        key = (frame_name, task_id)
        self._frames[key] = time.time()

    def frame_exit(self, frame_name: str, task_id: str) -> None:
        key = (frame_name, task_id)
        if key not in self._frames:
            logger.warning(
                f"TRACE: Exit without matching enter for {frame_name} [task={task_id}]"
            )
            return

        duration_ms = round((time.time() - self._frames[key]) * 1e3)
        del self._frames[key]

        msg = f"TRACE: {frame_name} [task={task_id}] completed in {duration_ms}ms"
        logger.info(msg)


# Tracy backend (placeholder)
class TracyBackend(TracingBackend):
    def init(self, app_name: str) -> None:
        raise NotImplementedError("Tracy tracing not yet implemented")

    def frame_enter(self, frame_name: str, task_id: str) -> None:
        raise NotImplementedError("Tracy tracing not yet implemented")

    def frame_exit(self, frame_name: str, task_id: str) -> None:
        raise NotImplementedError("Tracy tracing not yet implemented")


# Global tracing configuration
class TracingConfig:
    enabled: bool = True
    app_name: str = "ShortfinLLM"
    backend: TracingBackend = LoggingBackend()
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
            cls.backend = LoggingBackend()
        elif backend_name == "tracy":
            cls.backend = TracyBackend()
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
def trace_context(frame_name: str, task_id: str) -> Generator[None, None, None]:
    """Context manager for manual tracing of code blocks."""
    if not TracingConfig.is_enabled():
        yield
        return

    TracingConfig._ensure_initialized()
    TracingConfig.backend.frame_enter(frame_name, task_id)
    try:
        yield
    finally:
        TracingConfig.backend.frame_exit(frame_name, task_id)


@asynccontextmanager
async def async_trace_context(
    frame_name: str, task_id: str
) -> AsyncGenerator[None, None]:
    """Async context manager for manual tracing of code blocks."""
    if not TracingConfig.is_enabled():
        yield
        return

    TracingConfig._ensure_initialized()
    TracingConfig.backend.frame_enter(frame_name, task_id)
    try:
        yield
    finally:
        TracingConfig.backend.frame_exit(frame_name, task_id)


# Decorator for function tracing
def trace_fn(frame_name: Optional[str] = None):
    """
    Decorator for tracing function execution.

    Args:
        frame_name: Description of the function (defaults to function name)
    """

    def _decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped_fn_async(*args: Any, **kwargs: Any) -> Any:
            if not TracingConfig.is_enabled():
                return await func(*args, **kwargs)

            # Use function name if frame_name not provided
            fn_name = frame_name if frame_name is not None else func.__name__

            # Extract task ID if available
            task_id = "unknown"
            if len(args) > 0:
                if hasattr(args[0], "request_id"):
                    task_id = getattr(args[0], "request_id")
                elif (
                    hasattr(args[0], "exec_requests")
                    and len(getattr(args[0], "exec_requests", [])) > 0
                ):
                    first_req = getattr(args[0], "exec_requests")[0]
                    if hasattr(first_req, "request_id"):
                        task_id = getattr(first_req, "request_id")

            # Start tracing
            TracingConfig._ensure_initialized()
            TracingConfig.backend.frame_enter(fn_name, task_id)

            try:
                # Execute the function
                return await func(*args, **kwargs)
            finally:
                # End tracing
                TracingConfig.backend.frame_exit(fn_name, task_id)

        @functools.wraps(func)
        def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
            if not TracingConfig.is_enabled():
                return func(*args, **kwargs)

            # Use function name if frame_name not provided
            fn_name = frame_name if frame_name is not None else func.__name__

            # Extract task ID if available
            task_id = "unknown"
            if len(args) > 0:
                if hasattr(args[0], "request_id"):
                    task_id = getattr(args[0], "request_id")
                elif (
                    hasattr(args[0], "exec_requests")
                    and len(getattr(args[0], "exec_requests", [])) > 0
                ):
                    first_req = getattr(args[0], "exec_requests")[0]
                    if hasattr(first_req, "request_id"):
                        task_id = getattr(first_req, "request_id")

            # Start tracing
            TracingConfig._ensure_initialized()
            TracingConfig.backend.frame_enter(fn_name, task_id)

            try:
                # Execute the function
                return func(*args, **kwargs)
            finally:
                # End tracing
                TracingConfig.backend.frame_exit(fn_name, task_id)

        # Return async or sync wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return wrapped_fn_async
        return wrapped_fn

    return _decorator
