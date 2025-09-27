import json

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class DatasetTypes(Enum):
    """Enum for different types of datasets."""

    LOCAL = auto()


class AvailableDatasets(Enum):
    """Enum for different datasets used in integration tests."""

    ALL = auto()
    BASIC = auto()
    CHUNKED_PREFILL = auto()
    PREFIX_MATCHING = auto()


@dataclass
class DatasetRequest:
    """Class representing a dataset request."""

    dataset_type: DatasetTypes
    dataset: AvailableDatasets
    dataset_path: Path


class Dataset:
    """Class representing a dataset."""

    def __init__(self, request: DatasetRequest, batch_size: int = 1):
        self.request = request
        self.data = None
        self.batch_size = batch_size

    def _load_local(self):
        request = self.request
        path = request.dataset_path
        dataset = request.dataset.name.lower()

        data = None
        with open(request.dataset_path, "r") as f:
            data = json.load(f)

        if dataset not in data and dataset != AvailableDatasets.ALL.name.lower():
            raise KeyError(f"Dataset {dataset} not found in {path}")

        if dataset != AvailableDatasets.ALL.name.lower():
            self.data = data[dataset]
            return

        combined_data = {}
        for key in data:
            combined_data.update(data[key])

        self.data = combined_data

    @property
    def size(self):
        """Return the size of the dataset."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() before accessing data.")
        return len(self.data)

    def load(self):
        """Load the dataset based on the request type."""
        if self.request.dataset_type == DatasetTypes.LOCAL:
            return self._load_local()
        else:
            raise ValueError(f"Unsupported dataset type: {self.request.dataset_type}")

    def get_expected_generation(self, prompt: str) -> str:
        """Get the expected generation for a given prompt."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() before accessing data.")
        if prompt not in self.data:
            raise KeyError(f"Prompt '{prompt}' not found in dataset.")
        return self.data[prompt]

    def __iter__(self):
        self.load()
        data = self.data
        if data is None:
            raise ValueError("Dataset not loaded. Call load() before iterating.")

        for i in range(0, len(data.keys()), self.batch_size):
            yield {
                prompt: data[prompt]
                for prompt in list(data.keys())[i : i + self.batch_size]
            }
