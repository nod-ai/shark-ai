"""
This file supports using the Mooncake KVCache as part of the KVCache support in Shortfin.
"""

import json
import pathlib

import logging

from dataclasses import dataclass

import numpy as np

# from safetensors.torch import load as safetensors_load
# from safetensors.torch import save as safetensors_save
from safetensors.numpy import load as safetensors_load
from safetensors.numpy import save as safetensors_save

from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MooncakeConfig:
    """
    Configuration for connecting to mooncake master.
    """

    local_hostname: str = "localhost"
    metadata_server: str = "localhost:8080"
    global_segment_size: Optional[int] = 1024**3 * 3  # 3GB
    local_buffer_size: Optional[int] = 1024**3  # 1GB
    protocol: str = "tcp"
    device_name: str = ""
    master_server_address: str = "localhost:8080"

    @staticmethod
    def from_json(json_file: str) -> "MooncakeConfig":
        """
        Load MooncakeConfig from a JSON file.
        """
        file_path = pathlib.Path(json_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file {json_file} does not exist.")

        config = None
        with open(json_file, "r") as f:
            config_data = json.load(f)
            try:
                config = MooncakeConfig(**config_data)
            except KeyError as e:
                raise KeyError(f"Missing required configuration key: {e}")
        return config


class MooncakeStore:
    """
    MooncakeStore is a wrapper around the Mooncake KVCache.
    It provides methods to connect to the Mooncake master and perform operations on the KVCache.
    """

    def __init__(self, config: MooncakeConfig):

        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Mooncake is not installed. "
                "Please run "
                "pip3 install mooncake-transfer-engine "
                "to install the Mooncake package."
            ) from e

        try:
            self.config = config
            self.store = MooncakeDistributedStore()
            self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize MooncakeStore. "
                "Please check your Mooncake configuration."
            ) from e
        self._connected = True

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, no need to explicitly close it.
        pass

    def put(self, key: str, value: np.array) -> None:
        """
        Put a key-value pair into the Mooncake KVCache.
        """
        if not self._connected:
            raise RuntimeError("MooncakeStore is not connected.")
        # check if the key already exists
        is_exist = self.store.is_exist(key)
        if is_exist != 0:  # 0 means the key does not exist
            return None
        value_bytes = safetensors_save({"value": value})
        try:
            self.store.put(key, value_bytes)
            logger.info(f"Successfully put key: {key} into Mooncake KVCache.")
        except Exception as e:
            logger.error(f"Failed to put key: {key} into Mooncake KVCache. Error: {e}")
            raise e

    def put_int_list(self, key: str, value: list[int]) -> None:
        """
        Put a key-value pair into the Mooncake KVCache with a list of integers.
        This method is specifically for storing lists of integers as tensors.
        """
        value_array = np.array(value, dtype=np.int64)
        self.put(key, value_array)

    def get(self, key: str) -> Optional[np.array]:
        """
        Get a value from the Mooncake KVCache by key.
        """
        try:
            value_bytes = self.store.get(key)
        except TypeError as e:
            logger.error(f"Failed to get key: {key} from Mooncake KVCache. Error: {e}")
            raise TypeError("Mooncake Store Get Type Error.") from e

        if not value_bytes:
            return None

        if value_bytes:
            data = safetensors_load(value_bytes)
            value_array = data["value"]
            logger.info(f"Successfully retrieved key: {key} from Mooncake KVCache.")
            return value_array

    def get_int_list(self, key: str) -> Optional[list[int]]:
        """
        Get a list of integers from the Mooncake KVCache by key.
        This method is specifically for retrieving lists of integers stored as tensors.
        """
        value_array = self.get(key)
        if value_array is not None:
            return value_array.tolist()
        return None
