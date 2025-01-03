"""Handles server lifecycle and configuration."""
import json
import socket
from contextlib import closing
from dataclasses import dataclass, field
import subprocess
import time
import requests
from pathlib import Path
import sys
from typing import Optional

from .device_settings import DeviceSettings
from .model_management import ModelArtifacts


@dataclass
class ServerConfig:
    """Configuration for server instance."""

    port: int
    artifacts: ModelArtifacts
    device_settings: DeviceSettings

    # things we need to write to config
    prefix_sharing_algorithm: str = "none"


class ServerManager:
    """Manages server lifecycle and configuration."""

    @staticmethod
    def find_available_port() -> int:
        """Finds an available port for the server."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def __init__(self, config: ServerConfig):
        self.config = config

    def _write_config(self) -> Path:
        """Creates server config by extending the exported model config."""
        # TODO: eliminate this by moving prefix sharing algorithm to be a cmdline arg of server.py
        source_config_path = self.config.artifacts.config_path
        server_config_path = (
            source_config_path.parent
            / f"server_config_{self.config.prefix_sharing_algorithm}.json"
        )

        # Read the exported config as base
        with open(source_config_path) as f:
            config = json.load(f)
        config["paged_kv_cache"][
            "prefix_sharing_algorithm"
        ] = self.config.prefix_sharing_algorithm
        with open(server_config_path, "w") as f:
            json.dump(config, f)
        return server_config_path

    def start(self) -> subprocess.Popen:
        """Starts the server process."""
        config_path = self._write_config()
        cmd = [
            sys.executable,
            "-m",
            "shortfin_apps.llm.server",
            f"--tokenizer_json={self.config.artifacts.tokenizer_path}",
            f"--model_config={config_path}",
            f"--vmfb={self.config.artifacts.vmfb_path}",
            f"--parameters={self.config.artifacts.weights_path}",
            f"--port={self.config.port}",
        ]
        cmd.extend(self.config.device_settings.server_flags)
        process = subprocess.Popen(cmd)
        self._wait_for_server(timeout=10)
        return process

    def _wait_for_server(self, timeout: int = 10):
        """Waits for server to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                requests.get(f"http://localhost:{self.config.port}/health")
                return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise TimeoutError(f"Server failed to start within {timeout} seconds")
