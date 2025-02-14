"""Handles server lifecycle and configuration."""
import socket
from contextlib import closing, contextmanager
from dataclasses import dataclass
import subprocess
import time
import requests
import sys
from typing import Optional

from .device_settings import DeviceSettings
from .model_management import ModelArtifacts

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    @staticmethod
    def get_server_args(config: ServerConfig) -> list[str]:
        """Returns the command line arguments for the server."""
        return [
            f"--tokenizer_json={config.artifacts.tokenizer_path}",
            f"--model_config={config.artifacts.config_path}",
            f"--vmfb={config.artifacts.vmfb_path}",
            f"--parameters={config.artifacts.weights_path}",
            f"--port={config.port}",
            f"--prefix_sharing_algorithm={config.prefix_sharing_algorithm}",
        ] + config.device_settings.server_flags

    def start_full_fastapi_server(self) -> None:
        """Starts the server process."""
        if self.process is not None:
            raise RuntimeError("Server is already running")

        self.port = self.find_available_port()

        cmd = [sys.executable, "-m", "shortfin_apps.llm.server"]
        cmd.extend(self.get_server_args(self.config))

        self.process = subprocess.Popen(cmd)
        self.wait_for_ready()

    @contextmanager
    def start_generate_service(
        model_artifacts, request
    ) -> "shortfin_apps.llm.components.service.GenerateService":
        """
        like server, but no fastapi,

        this yields a service object that gives access to shortfin while bypassing fastapi

        use like so:

        ```
        with instance.start_generate_service(model_artifacts, request) as service:
            # run tests with service
            ...
        ```
        """

        model_config = model_artifacts.model_config

        server_config = ServerConfig(
            artifacts=model_artifacts,
            device_settings=model_config.device_settings,
            prefix_sharing_algorithm=request.param.get("prefix_sharing", "none"),
        )

        from shortfin_apps.llm import server as server_module

        argv = ServerInstance.get_server_args(server_config)
        args = server_module.parse_args(argv)
        server_module.sysman = server_module.configure(args)
        sysman = server_module.sysman
        services = server_module.services
        # sysman.start()
        try:
            for service_name, service in sysman.services.items():
                logging.info("Initializing service '%s': %r", service_name, service)
                service.start()
        except:
            sysman.shutdown()
            raise
        yield sysman.services["default"]
        try:
            for service_name, service in services.items():
                logging.info("Shutting down service '%s'", service_name)
                service.shutdown()
        finally:
            sysman.shutdown()

    def wait_for_ready(self, timeout: int = 30) -> None:
        """Waits for server to be ready and responding to health checks."""
        if self.port is None:
            raise RuntimeError("Server hasn't been started")

        start = time.time()
        while time.time() - start < timeout:
            try:
                requests.get(f"http://localhost:{self.port}/health")
                return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise TimeoutError(f"Server failed to start within {timeout} seconds")

    def stop(self) -> None:
        """Stops the server process."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.port = None
