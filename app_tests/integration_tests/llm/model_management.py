"""Module for managing model artifacts through various processing stages."""
import logging
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from enum import Enum, auto

from sharktank.utils.hf_datasets import Dataset, RemoteFile, get_dataset

logger = logging.getLogger(__name__)


class AccuracyValidationException(RuntimeError):
    """Exception raised when accuracy validation fails."""

    pass


class ModelSource(Enum):
    HUGGINGFACE = auto()
    LOCAL = auto()
    AZURE = auto()


@dataclass
class AzureConfig:
    """Configuration for Azure blob storage downloads."""

    account_name: str
    container_name: str
    blob_path: str
    auth_mode: str = "key"


@dataclass
class ModelConfig:
    """Configuration for model source and settings."""

    model_file: str
    tokenizer_id: str
    batch_sizes: Tuple[int, ...]
    device_settings: "DeviceSettings"
    source: ModelSource
    dataset_name: Optional[str] = None  # Name of the dataset in hf_datasets.py
    repo_id: Optional[str] = None
    local_path: Optional[Path] = None
    azure_config: Optional[AzureConfig] = None

    def __post_init__(self):
        if self.source == ModelSource.HUGGINGFACE:
            if not (self.dataset_name or self.repo_id):
                raise ValueError(
                    "Either dataset_name or repo_id required for HuggingFace models"
                )
        elif self.source == ModelSource.LOCAL and not self.local_path:
            raise ValueError("local_path required for local models")
        elif self.source == ModelSource.AZURE and not self.azure_config:
            raise ValueError("azure_config required for Azure models")


@dataclass
class ModelArtifacts:
    """Container for all paths related to model artifacts."""

    weights_path: Path
    tokenizer_path: Path
    mlir_path: Path
    vmfb_path: Path
    config_path: Path


class ModelStageManager:
    """Manages different stages of model processing with caching behavior."""

    def __init__(self, base_dir: Path, config: ModelConfig):
        self.base_dir = base_dir
        self.config = config
        self.model_dir = self._get_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self) -> Path:
        """Creates and returns appropriate model directory based on source."""
        if self.config.source == ModelSource.HUGGINGFACE:
            if self.config.dataset_name:
                return self.base_dir / self.config.dataset_name.replace("/", "_")
            return self.base_dir / self.config.repo_id.replace("/", "_")
        elif self.config.source == ModelSource.LOCAL:
            return self.base_dir / "local" / self.config.local_path.stem
        elif self.config.source == ModelSource.AZURE:
            return (
                self.base_dir
                / "azure"
                / self.config.azure_config.blob_path.replace("/", "_")
            )
        raise ValueError(f"Unsupported model source: {self.config.source}")

    def _download_from_huggingface(self) -> Path:
        """Downloads model from HuggingFace using hf_datasets.py."""
        model_path = self.model_dir / self.config.model_file
        if not model_path.exists():
            if self.config.dataset_name:
                logger.info(
                    f"Downloading model {self.config.dataset_name} using hf_datasets"
                )
                dataset = get_dataset(self.config.dataset_name)
                downloaded_files = dataset.download(local_dir=self.model_dir)

                # Find the model file in downloaded files
                for file_id, paths in downloaded_files.items():
                    for path in paths:
                        if path.name == self.config.model_file:
                            return path

                raise ValueError(
                    f"Model file {self.config.model_file} not found in dataset {self.config.dataset_name}"
                )
            else:
                logger.info(f"Downloading model {self.config.repo_id} from HuggingFace")
                # Create a temporary dataset for direct repo downloads
                remote_file = RemoteFile(
                    file_id="model",
                    repo_id=self.config.repo_id,
                    filename=self.config.model_file,
                )
                downloaded_paths = remote_file.download(local_dir=self.model_dir)
                return downloaded_paths[0]

        return model_path

    def _copy_from_local(self) -> Path:
        """Copies model from local filesystem."""
        import shutil

        model_path = self.model_dir / self.config.model_file
        if not model_path.exists():
            logger.info(f"Copying local model from {self.config.local_path}")
            shutil.copy2(self.config.local_path, model_path)
        return model_path

    def _download_from_azure(self) -> Path:
        """Downloads model from Azure blob storage."""
        model_path = self.model_dir / self.config.model_file
        if not model_path.exists():
            logger.info(
                f"Downloading model from Azure blob storage: {self.config.azure_config.blob_path}"
            )
            subprocess.run(
                [
                    "az",
                    "storage",
                    "blob",
                    "download",
                    "--account-name",
                    self.config.azure_config.account_name,
                    "--container-name",
                    self.config.azure_config.container_name,
                    "--name",
                    self.config.azure_config.blob_path,
                    "--file",
                    str(model_path),
                    "--auth-mode",
                    self.config.azure_config.auth_mode,
                ],
                check=True,
            )
        return model_path

    def prepare_tokenizer(self) -> Path:
        """Downloads and prepares tokenizer using hf_datasets.py when possible."""
        tokenizer_path = self.model_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            # First try to get tokenizer from dataset if available
            if self.config.dataset_name:
                dataset = get_dataset(self.config.dataset_name)
                downloaded_files = dataset.download(local_dir=self.model_dir)

                # Look for tokenizer files in downloaded files
                for file_id, paths in downloaded_files.items():
                    for path in paths:
                        if path.name == "tokenizer.json":
                            return path

            # Fall back to downloading from transformers if not found in dataset
            logger.info(
                f"Downloading tokenizer {self.config.tokenizer_id} using transformers"
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_id)
            tokenizer.save_pretrained(self.model_dir)

        return tokenizer_path

    def export_model(self, weights_path: Path) -> Tuple[Path, Path]:
        """Exports model to MLIR format."""
        bs_string = ",".join(map(str, self.config.batch_sizes))
        mlir_path = self.model_dir / "model.mlir"
        config_path = self.model_dir / "config.json"

        logger.info(
            "Exporting model with following settings:\n"
            f"  MLIR Path: {mlir_path}\n"
            f"  Config Path: {config_path}\n"
            f"  Batch Sizes: {bs_string}"
        )

        subprocess.run(
            [
                "python",
                "-m",
                "sharktank.examples.export_paged_llm_v1",
                "--block-seq-stride=16",
                f"--{weights_path.suffix.strip('.')}-file={weights_path}",
                f"--output-mlir={mlir_path}",
                f"--output-config={config_path}",
                f"--bs={bs_string}",
            ],
            check=True,
        )

        logger.info(f"Model successfully exported to {mlir_path}")
        return mlir_path, config_path

    def compile_model(self, mlir_path: Path) -> Path:
        """Compiles model to VMFB format."""
        vmfb_path = self.model_dir / "model.vmfb"
        logger.info(f"Compiling model to {vmfb_path}")

        compile_command = [
            "iree-compile",
            str(mlir_path),
            "-o",
            str(vmfb_path),
        ]
        compile_command.extend(self.config.device_settings.compile_flags)

        subprocess.run(compile_command, check=True)
        logger.info(f"Model successfully compiled to {vmfb_path}")
        return vmfb_path


class ModelProcessor:
    """Main interface for processing models through all stages."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def process_model(self, config: ModelConfig) -> ModelArtifacts:
        """Process model through all stages and return paths to all artifacts."""
        manager = ModelStageManager(self.base_dir, config)

        # Stage 1: Download weights and tokenizer (cached)
        if config.source == ModelSource.HUGGINGFACE:
            weights_path = manager._download_from_huggingface()
        elif config.source == ModelSource.LOCAL:
            weights_path = manager._copy_from_local()
        elif config.source == ModelSource.AZURE:
            weights_path = manager._download_from_azure()
        else:
            raise ValueError(f"Unsupported model source: {config.source}")

        tokenizer_path = manager.prepare_tokenizer()

        # Stage 2: Export model (fresh every time)
        mlir_path, config_path = manager.export_model(weights_path)

        # Stage 3: Compile model (fresh every time)
        vmfb_path = manager.compile_model(mlir_path)

        return ModelArtifacts(
            weights_path=weights_path,
            tokenizer_path=tokenizer_path,
            mlir_path=mlir_path,
            vmfb_path=vmfb_path,
            config_path=config_path,
        )
