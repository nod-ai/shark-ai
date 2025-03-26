"""Handles server lifecycle and configuration."""
import os
from datetime import datetime
import socket
from contextlib import closing
from dataclasses import dataclass
import subprocess
import time
import requests
import sys
from typing import Optional

from .device_settings import DeviceSettings
from .model_management import ModelArtifacts
from shortfin_apps.llm.components.service import GenerateService
from contextlib import contextmanager

from logging import getLogger

logger = getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for server instance."""

    artifacts: ModelArtifacts
    device_settings: DeviceSettings
    prefix_sharing_algorithm: str = "none"


class ServerInstance:
    """An instance of the shortfin llm inference server.

    Example usage:

    ```
        from shortfin_apps.llm.server_management import ServerInstance, ServerConfig
        # Create and start server
        server = Server(config=ServerConfig(
            artifacts=model_artifacts,
            device_settings=device_settings,
            prefix_sharing_algorithm="none"
        ))

        server.start()  # This starts the server and waits for it to be ready

        # Use the server
        print(f"Server running on port {server.port}")

        # Cleanup when done
        server.stop()
    ```
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None

    @staticmethod
    def find_available_port() -> int:
        """Finds an available port for the server."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def get_server_args(self) -> list[str]:
        """Returns the command line arguments to start the server."""
        argv = [
            f"--tokenizer_json={self.config.artifacts.tokenizer_path}",
            f"--model_config={self.config.artifacts.config_path}",
            f"--vmfb={self.config.artifacts.vmfb_path}",
            f"--parameters",
            str(self.config.artifacts.weights_path),
            *(str(path) for path in (self.config.artifacts.shard_paths or [])),
            f"--port={self.port}",
            f"--prefix_sharing_algorithm={self.config.prefix_sharing_algorithm}",
        ]
        argv.extend(self.config.device_settings.server_flags)
        return argv

    @contextmanager
    def start_service_only(self) -> GenerateService:
        """Starts a server with only the shortfin_apps.llm.components.serivce.GenerateService."""

        argv = self.get_server_args()
        from shortfin_apps.llm.server import parse_args

        args = parse_args(argv)
        if args.tokenizer_config_json is None:
            # this is only used for the EOS token
            inferred_tokenizer_config_path = args.tokenizer_json.with_name(
                args.tokenizer_json.stem + "_config.json"
            )
        args.tokenizer_config_json = inferred_tokenizer_config_path

        from shortfin_apps.llm.components.lifecycle import ShortfinLlmLifecycleManager

        lifecycle_manager = ShortfinLlmLifecycleManager(args)

        with lifecycle_manager:
            yield lifecycle_manager.services["default"]

    def start(self, log_dir="/tmp") -> None:
        """Starts the server process."""
        if self.process is not None:
            raise RuntimeError("Server is already running")

        self.port = self.find_available_port()

        cmd = [
            sys.executable,
            "-m",
            "shortfin_apps.llm.server",
        ] + self.get_server_args()
        logger.info("Starting server with command: %s", " ".join(cmd))

        # Create log file for capturing output
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"server_{timestamp}_{self.port}.log")

        logger.info("Server logs: %s", self.log_path)

        # Log file will be automatically closed when process terminates
        self.process = subprocess.Popen(
            cmd,
            stdout=open(
                self.log_path, "w"
            ),  # File handle will be garbage collected with process
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            close_fds=True,  # Ensures file descriptors are closed when process exits
        )

        try:
            self.wait_for_ready()
        except Exception as e:
            # Log path for easier debugging on failure
            logger.error("Server failed to start, check logs at: %s", self.log_path)
            # Print the tail of the log for immediate debugging
            self._dump_log_tail()
            raise

    def wait_for_ready(self, timeout: int = 180) -> None:
        """Waits for server to be ready and responding to health checks."""
        if self.port is None:
            raise RuntimeError("Server hasn't been started")

        start = time.time()
        while time.time() - start < timeout:
            try:
                requests.get(f"http://localhost:{self.port}/health")
                return
            except requests.exceptions.ConnectionError as e:
                logger.info("While attempting to server,")
                logger.info("Encountered connection error %s", e)
                time.sleep(1)
        raise TimeoutError(f"Server failed to start within {timeout} seconds")

    def _dump_log_tail(self, lines=20):
        """Dumps the tail of the log file for debugging."""
        if hasattr(self, "log_path") and os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    log_content = f.readlines()
                    tail = (
                        log_content[-lines:]
                        if len(log_content) > lines
                        else log_content
                    )
                    if tail:
                        logger.error("Last %d lines of server log:", len(tail))
                        for line in tail:
                            logger.error("  %s", line.rstrip())
            except Exception as e:
                logger.error("Failed to read log file: %s", str(e))

    def stop(self) -> None:
        """Stops the server process."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.port = None
