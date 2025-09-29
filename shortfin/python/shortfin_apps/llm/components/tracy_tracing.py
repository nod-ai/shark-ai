import functools
import logging
import importlib.util
from typing import Dict, Optional

from .tracing import BaseTracingBackend

# Configure logger
logger = logging.getLogger("shortfin-llm.tracy-tracing")

# Check if pytracy is available
_has_pytracy = importlib.util.find_spec("pytracy") is not None
if _has_pytracy:
    import pytracy


class TracyTracingBackend(BaseTracingBackend):
    """Tracy profiler integration for Shortfin LLM.

    This backend integrates with Tracy profiler (https://github.com/wolfpld/tracy)
    using pytracy (https://github.com/nschloe/pytracy/).

    To use this backend:
    1. Install pytracy: pip install pytracy
    2. Use set_backend("tracy") in your application
    """

    def __init__(self):
        # Check if Tracy is available
        if not _has_pytracy:
            logger.warning(
                "Tracy backend requested but pytracy module not available. "
                "Install with: pip install pytracy"
            )

        # Frame tracking - maps (frame_name, task_id) to Tracy zones
        self._zones: Dict[tuple, Optional[pytracy.Zone]] = {}
        self._initialized = False

    def init(self, app_name: str) -> None:
        """Initialize Tracy backend with the application name."""
        if not _has_pytracy:
            logger.warning("Skipping Tracy initialization - pytracy not available")
            return

        # Only initialize once
        if self._initialized:
            return

        # Start profiling with the app name
        pytracy.initialize()
        pytracy.frame_mark_start(app_name)
        self._initialized = True
        logger.info(f"Tracy profiling initialized for application: {app_name}")

    def frame_enter(self, frame_name: str, task_id: str) -> None:
        """Start a Tracy profiling zone."""
        if not _has_pytracy or not self._initialized:
            return

        # Create a zone key
        key = (frame_name, task_id)

        # Create zone name with task ID
        zone_name = f"{frame_name} [task={task_id}]"

        # Start a new zone
        try:
            zone = pytracy.Zone(zone_name)
            self._zones[key] = zone
        except Exception as e:
            logger.error(f"Failed to create Tracy zone: {e}")
            self._zones[key] = None

    def frame_exit(self, frame_name: str, task_id: str) -> None:
        """End a Tracy profiling zone."""
        if not _has_pytracy or not self._initialized:
            return

        key = (frame_name, task_id)

        # Get the zone if it exists
        zone = self._zones.pop(key, None)

        # End the zone if it exists
        if zone is not None:
            try:
                zone.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Failed to end Tracy zone: {e}")

    def __del__(self):
        """Clean up Tracy when the backend is destroyed."""
        if _has_pytracy and self._initialized:
            try:
                # End any remaining zones
                for zone in self._zones.values():
                    if zone is not None:
                        zone.__exit__(None, None, None)
                self._zones.clear()

                # End frame marking
                pytracy.frame_mark_end()
                pytracy.shutdown()
            except Exception as e:
                logger.error(f"Error during Tracy shutdown: {e}")
