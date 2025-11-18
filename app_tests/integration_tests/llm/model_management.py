"""Module for managing model artifacts through various processing stages."""

import hashlib
import logging
import pytest
import subprocess
import tempfile
import urllib.request
import zipfile

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum, auto

from amdsharktank.utils.hf_datasets import Dataset, RemoteFile, get_dataset

from .device_settings import DeviceSettings, get_device_settings_by_name

logger = logging.getLogger(__name__)


def get_llama_cpp_path() -> Path:
    """Downloads and extracts llama.cpp if needed, returns path to installation."""
    temp_base = Path(tempfile.gettempdir()) / "amdsharktank_llamacpp"
    llama_cpp_dir = temp_base / "llama.cpp-b4696"

    if not llama_cpp_dir.exists():
        temp_base.mkdir(parents=True, exist_ok=True)
        zip_path = temp_base / "llama.cpp.zip"

        logger.info("Downloading llama.cpp...")
        urllib.request.urlretrieve(
            "https://github.com/ggerganov/llama.cpp/archive/refs/tags/b4696.zip",
            zip_path,
        )
        logger.info("Extracting llama.cpp...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_base)
        zip_path.unlink()
        logger.info(f"llama.cpp installed at {llama_cpp_dir}")
    return llama_cpp_dir


class AccuracyValidationException(RuntimeError):
    """Custom exception for accuracy validation failures."""

    def __init__(
        self,
        message: str = None,
        expected: str = "[[expected generation output not provided]]",
        actual: str = "[[actual generation output not provided]]",
    ):
        self.expected = expected
        self.actual = actual
        self.message = (
            message
            or f"Output validation failed.\nExpected: {expected}\nActually: {actual}"
        )
        super().__init__(self.message)


class ModelSource(Enum):
    HUGGINGFACE_FROM_GGUF = auto()
    LOCAL = auto()
    AZURE = auto()
    HUGGINGFACE_FROM_SAFETENSORS = auto()
    LOCAL_IRPA = auto()


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

    name: str
    model_file: str
    tokenizer_id: str
    batch_sizes_prefill: Tuple[int, ...]
    batch_sizes_decode: Tuple[int, ...]
    device_settings: "DeviceSettings"
    source: ModelSource
    dataset_name: Optional[str] = None  # Name of the dataset in hf_datasets.py
    repo_id: Optional[str] = None
    local_path: Optional[Path] = None
    azure_config: Optional[AzureConfig] = None
    tensor_parallelism_size: Optional[
        int
    ] = None  # Number of shards for tensor parallelism
    top_k: Optional[int] = None
    has_prefill_position: Optional[bool] = None
    irpa_path: Optional[Path] = None
    tokenizer_path: Optional[Path] = None
    block_seq_stride: int = 16

    def __post_init__(self):
        if self.source == ModelSource.HUGGINGFACE_FROM_GGUF:
            if not (self.dataset_name or self.repo_id):
                raise ValueError(
                    "Either dataset_name or repo_id required for HuggingFace models"
                )
        elif self.source == ModelSource.AZURE and not self.azure_config:
            raise ValueError("azure_config required for Azure models")
        elif self.source == ModelSource.HUGGINGFACE_FROM_SAFETENSORS:
            if not self.dataset_name:
                raise ValueError(
                    "dataset_name required for HUGGINGFACE_FROM_SAFETENSORS models"
                )

    @staticmethod
    def get(name, tp_size=None):
        """Get a model config by name, with optional tensor parallelism.

        Args:
            name: Base model name
            tp_size: Optional tensor parallelism size
            batch_sizes: Optional tuple of batch sizes to support

        Returns:
            ModelConfig: The requested model configuration

        Raises:
            KeyError: If the base model name is not found in the predefined models
        """
        # Check if the base model exists in predefined models
        if name not in _PREDEFINED_MODELS:
            # Try to parse a model pattern like "model_name_tp4"
            import re

            tp_match = re.match(r"(.+)_tp(\d+)$", name)
            if tp_match:
                base_name, tp_size_str = tp_match.groups()
                if base_name in _PREDEFINED_MODELS:
                    return ModelConfig.get(base_name, int(tp_size_str))
            raise KeyError(
                f"Model '{name}' not found. Available models: {list(_PREDEFINED_MODELS.keys())}"
            )

        # Get the base model config
        base_config = _PREDEFINED_MODELS[name]

        if tp_size is None:
            return base_config

        # Set tp and batch size
        return ModelConfig(
            name=name,
            source=base_config.source,
            repo_id=base_config.repo_id,
            dataset_name=base_config.dataset_name,
            model_file=base_config.model_file,
            tokenizer_id=base_config.tokenizer_id,
            batch_sizes_prefill=base_config.batch_sizes_prefill,
            batch_sizes_decode=base_config.batch_sizes_decode,
            device_settings=base_config.device_settings,
            local_path=base_config.local_path,
            azure_config=base_config.azure_config,
            tensor_parallelism_size=tp_size,
        )


# Dictionary of predefined base model configurations
_PREDEFINED_MODELS = {
    "llama3.1_8b": ModelConfig(
        name="llama3.1_8b",
        source=ModelSource.HUGGINGFACE_FROM_GGUF,
        repo_id="SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
        model_file="meta-llama-3.1-8b-instruct.f16.gguf",
        tokenizer_id="NousResearch/Meta-Llama-3.1-8B",
        batch_sizes_prefill=(4,),
        batch_sizes_decode=(4,),
        device_settings=None,
    ),
    "local_meta_llama3.1_8b_instruct": ModelConfig(
        name="local_meta_llama3.1_8b_instruct",
        source=ModelSource.LOCAL,
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        model_file="meta-llama-3.1-8b-instruct-fp16.gguf",
        tokenizer_id="meta-llama/Llama-3.1-8B-Instruct",
        batch_sizes_prefill=(8,),
        batch_sizes_decode=(32,),
        block_seq_stride=32,
        device_settings=None,
    ),
    "local_meta_llama3.1_8b_instruct_chunked": ModelConfig(
        name="local_meta_llama3.1_8b_instruct_chunked",
        source=ModelSource.LOCAL,
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        model_file="meta-llama-3.1-8b-instruct-fp16.gguf",
        tokenizer_id="meta-llama/Llama-3.1-8B-Instruct",
        batch_sizes_prefill=(8,),
        batch_sizes_decode=(32,),
        block_seq_stride=32,
        device_settings=None,
        has_prefill_position=True,
    ),
    "tinystories_llama2_25m": ModelConfig(
        name="tinystories_llama2_25m",
        source=ModelSource.HUGGINGFACE_FROM_SAFETENSORS,
        dataset_name="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        model_file="model.irpa",  # This will be the final converted file name
        tokenizer_id="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        batch_sizes_prefill=(4, 8),
        batch_sizes_decode=(4, 8),
        device_settings=None,
    ),
    "tinystories_llama2_25m_has_prefill_position": ModelConfig(
        name="tinystories_llama2_25m_has_prefill_position",
        source=ModelSource.HUGGINGFACE_FROM_SAFETENSORS,
        dataset_name="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        model_file="model.irpa",  # This will be the final converted file name
        tokenizer_id="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        batch_sizes_prefill=(4, 8),
        batch_sizes_decode=(4, 8),
        device_settings=None,
        has_prefill_position=True,
    ),
    "tinystories_llama2_25m_gpu_argmax": ModelConfig(
        name="tinystories_llama2_25m_gpu_argmax",
        source=ModelSource.HUGGINGFACE_FROM_SAFETENSORS,
        dataset_name="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        model_file="model.irpa",  # This will be the final converted file name
        tokenizer_id="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        batch_sizes_prefill=(4,),
        batch_sizes_decode=(4,),
        device_settings=None,
        top_k=1,
    ),
    "tinystories_llama2_25m_gpu_topk_k4": ModelConfig(
        name="tinystories_llama2_25m_gpu_topk_k4",
        source=ModelSource.HUGGINGFACE_FROM_SAFETENSORS,
        dataset_name="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        model_file="model.irpa",  # This will be the final converted file name
        tokenizer_id="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
        batch_sizes_prefill=(4,),
        batch_sizes_decode=(4,),
        device_settings=None,
        top_k=4,
    ),
}


@dataclass
class ModelArtifacts:
    """Container for all paths related to model artifacts."""

    weights_path: (
        Path  # Main weights file (the .irpa without .rankX for sharded models)
    )
    tokenizer_path: Path
    mlir_path: Path
    vmfb_path: Path
    config_path: Path
    model_config: (
        ModelConfig  # config that was originally used to generate these artifacts
    )
    shard_paths: Optional[
        list[Path]
    ] = None  # Paths to sharded weight files (model_name.rank\d+.irpa)
    irpa_path: Optional[Path] = None  # Path to the .irpa file if applicable


class ModelStageManager:
    """Manages different stages of model processing with caching behavior."""

    def __init__(self, base_dir: Path, config: ModelConfig):
        self.base_dir = base_dir
        self.config = config
        self.model_dir = self._get_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self) -> Path:
        """Creates and returns appropriate model directory based on source."""
        if self.config.source == ModelSource.HUGGINGFACE_FROM_GGUF:
            if self.config.dataset_name:
                return self.base_dir / self.config.dataset_name.replace("/", "_")
            return self.base_dir / self.config.repo_id.replace("/", "_")
        elif self.config.source == ModelSource.LOCAL:
            return self.base_dir / "local" / self.config.name
        elif self.config.source == ModelSource.AZURE:
            return (
                self.base_dir
                / "azure"
                / self.config.azure_config.blob_path.replace("/", "_")
            )
        elif self.config.source == ModelSource.HUGGINGFACE_FROM_SAFETENSORS:
            return self.base_dir / self.config.dataset_name.replace("/", "_")
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

    def _download_and_convert_from_huggingface(self) -> Path:
        """Downloads model from HuggingFace and converts through GGUF to IRPA."""
        irpa_path = self.model_dir / "model.irpa"

        if not irpa_path.exists():
            logger.info(
                f"Processing model `{self.config.dataset_name}` from HuggingFace through GGUF to IRPA"
            )

            # Step 1: Download from HuggingFace
            hf_model_path = self.model_dir / "model_hf_repo_clone"
            if not hf_model_path.exists():
                logger.info(
                    f"Downloading model from HuggingFace: `{self.config.dataset_name}`"
                )
                dataset = get_dataset(self.config.dataset_name)
                downloaded_files = dataset.download(local_dir=self.model_dir)

            # Step 2: Convert to GGUF
            gguf_path = self.model_dir / "model.gguf"
            if not gguf_path.exists():
                logger.info("Converting model to GGUF format")
                subprocess.run(
                    [
                        "python",
                        get_llama_cpp_path() / "convert_hf_to_gguf.py",
                        self.model_dir,
                        "--outfile",
                        str(gguf_path),
                        "--outtype",
                        "f32",
                    ],
                    check=True,
                )

            # Step 3: Convert to IRPA
            logger.info("Converting GGUF to IRPA format")
            subprocess.run(
                [
                    "python",
                    "-m",
                    "amdsharktank.tools.dump_gguf",
                    f"--gguf-file={gguf_path}",
                    f"--output-irpa={str(irpa_path)}",
                ],
                check=True,
            )

            # Cleanup intermediate files if desired
            # shutil.rmtree(hf_model_path)
            # gguf_path.unlink()

        return irpa_path

    def _copy_from_local(self) -> Path:
        """Copies model from local filesystem."""
        model_path = self.model_dir / self.config.model_file
        if not model_path.exists():
            import shutil

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

    def shard_model(self, weights_path: Path) -> Tuple[Path, list[Path]]:
        """Shards model using tensor parallelism if configured."""
        if not self.config.tensor_parallelism_size:
            return weights_path, None

        # Determine device type from compile flags
        device_type = "cpu"  # Default to CPU
        compile_flags = self.config.device_settings.compile_flags
        for flag in compile_flags:
            if "hip" in flag.lower():
                device_type = "hip"  # Use "hip" for AMD GPU device
                break

        logger.info(
            f"Sharding model with tensor parallelism size {self.config.tensor_parallelism_size} "
            f"for device type: {device_type}"
        )

        base_name = weights_path.stem
        output_base = self.model_dir / f"{base_name}.sharded"
        output_irpa = output_base.with_suffix(".irpa")

        shard_cmd = [
            "python",
            "-m",
            "amdsharktank.examples.sharding.shard_llm_dataset",
            f"--{weights_path.suffix.strip('.')}-file={weights_path}",
            f"--output-irpa={output_irpa}",
            f"--tensor-parallelism-size={self.config.tensor_parallelism_size}",
        ]

        logger.info(f"Running sharding command: {' '.join(shard_cmd)}")

        try:
            result = subprocess.run(
                shard_cmd, check=True, capture_output=True, text=True
            )
            logger.info(f"Sharding succeeded")
        except subprocess.CalledProcessError as e:
            logger.error(f"Sharding failed with code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

        shard_paths = [
            output_base.with_suffix(f".rank{i}.irpa")
            for i in range(self.config.tensor_parallelism_size)
        ]

        logger.info(f"Model successfully sharded into {len(shard_paths)} shards")
        return output_irpa, shard_paths

    def export_model(
        self, weights_path: Path, block_seq_stride: int
    ) -> Tuple[Path, Path]:
        """Exports model to MLIR format."""
        bs_string_prefill = ",".join(map(str, self.config.batch_sizes_prefill))
        bs_string_decode = ",".join(map(str, self.config.batch_sizes_decode))
        mlir_path = self.model_dir / "model.mlir"
        config_path = self.model_dir / "config.json"
        logger.info(
            "Exporting model with following settings:\n"
            f"  MLIR Path: {mlir_path}\n"
            f"  Config Path: {config_path}\n"
            f"  Batch Sizes Prefill: {bs_string_prefill}"
            f"  Batch Sizes Decode: {bs_string_decode}"
        )

        if self.config.tensor_parallelism_size:
            weights_path = weights_path.with_suffix(".irpa")

        export_cmd = [
            "python",
            "-m",
            "amdsharktank.examples.export_paged_llm_v1",
            f"--block-seq-stride={block_seq_stride}",
            f"--{weights_path.suffix.strip('.')}-file={weights_path}",
            f"--output-mlir={mlir_path}",
            f"--output-config={config_path}",
            f"--bs-prefill={bs_string_prefill}",
            f"--bs-decode={bs_string_decode}",
        ]

        if self.config.tensor_parallelism_size:
            export_cmd.append(
                f"--tensor-parallelism-size={self.config.tensor_parallelism_size}"
            )

        if self.config.top_k is not None:
            export_cmd.append(f"--top-k={self.config.top_k}")

        if self.config.has_prefill_position is not None:
            export_cmd.append("--has-prefill-position")

        logger.info(f"Running export command: {' '.join(export_cmd)}")

        try:
            result = subprocess.run(
                export_cmd, check=True, capture_output=True, text=True
            )
            logger.info(f"Export succeeded.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Export failed with code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

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

        logger.info(f"Running compiler command: {' '.join(compile_command)}")
        try:
            result = subprocess.run(
                compile_command, check=True, capture_output=True, text=True
            )
            logger.info(f"Compilation succeeded")
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation failed with code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

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
        tokenizer_path = None
        if config.source == ModelSource.HUGGINGFACE_FROM_GGUF:
            weights_path = manager._download_from_huggingface()
        elif config.source == ModelSource.LOCAL:
            weights_path = config.irpa_path
            tokenizer_path = config.tokenizer_path
        elif config.source == ModelSource.AZURE:
            weights_path = manager._download_from_azure()
        elif config.source == ModelSource.HUGGINGFACE_FROM_SAFETENSORS:
            weights_path = manager._download_and_convert_from_huggingface()
        else:
            raise ValueError(f"Unsupported model source: {config.source}")

        if tokenizer_path is None:
            tokenizer_path = manager.prepare_tokenizer()

        # Stage 1.5: Shard model if tensor parallelism is configured
        shard_paths = None
        if config.tensor_parallelism_size:
            weights_path, shard_paths = manager.shard_model(weights_path)

        # Stage 2: Export model (fresh every time)
        mlir_path, config_path = manager.export_model(
            weights_path, config.block_seq_stride
        )

        # Stage 3: Compile model (fresh every time)
        vmfb_path = manager.compile_model(mlir_path)

        return ModelArtifacts(
            weights_path=weights_path,
            tokenizer_path=tokenizer_path,
            mlir_path=mlir_path,
            vmfb_path=vmfb_path,
            config_path=config_path,
            model_config=config,
            shard_paths=shard_paths,
            irpa_path=config.irpa_path,
        )


@dataclass
class ModelBatcherConfig:
    model_configs: List[ModelConfig]
    test_device: str
    cache_dir: Path
    irpa_path: Optional[Path] = None
    tokenizer_path: Optional[Path] = None


class ModelBatcher:
    def _validate_model_config(
        self,
        model_config: ModelConfig,
        test_device: str,
        irpa_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
    ):
        if test_device == "cpu" and model_config.tensor_parallelism_size is not None:
            pytest.skip("Skipping CPU tests with tensor parallelism")

        if test_device == "cpu" and model_config.has_prefill_position is not None:
            pytest.skip(
                "Skipping CPU tests with prefill position due to compilation error"
            )

        if model_config.source == ModelSource.LOCAL:
            if irpa_path is None:
                pytest.fail("IRPA path must be specified for LOCAL models")
            if tokenizer_path is None:
                pytest.fail("Tokenizer path must be specified for LOCAL models")

            model_config.irpa_path = Path(irpa_path)
            model_config.tokenizer_path = Path(tokenizer_path)

        return model_config

    def _prepare_model(
        self, model_config, irpa_path, tokenizer_path, test_device, cache_dir
    ) -> ModelArtifacts | ModelConfig:
        model_config = self._validate_model_config(
            model_config, test_device, irpa_path, tokenizer_path
        )

        settings_key = test_device
        if (
            model_config.tensor_parallelism_size is not None
            and model_config.tensor_parallelism_size > 1
        ):
            settings_key += f"_tp{model_config.tensor_parallelism_size}"
        model_config.device_settings = get_device_settings_by_name(settings_key)

        cache_key = hashlib.md5(str(model_config).encode()).hexdigest()
        model_dir = cache_dir / cache_key
        if model_dir.exists():
            return ModelArtifacts(
                weights_path=model_dir / model_config.model_file,
                tokenizer_path=model_dir / "tokenizer.json",
                mlir_path=model_dir / "model.mlir",
                vmfb_path=model_dir / "model.vmfb",
                config_path=model_dir / "config.json",
                model_config=model_config,
            )

        return model_config

    def _process_model(
        self, model_config: ModelConfig, cache_dir: Path
    ) -> ModelArtifacts:
        """
        Worker function executed in a separate process.
        Creates artifacts for a single model_config in cache_dir (if not already cached).
        """
        cache_key = hashlib.md5(str(model_config).encode()).hexdigest()
        model_dir = cache_dir / cache_key

        if model_dir.exists():
            return ModelArtifacts(
                weights_path=model_dir / model_config.model_file,
                tokenizer_path=model_dir / "tokenizer.json",
                mlir_path=model_dir / "model.mlir",
                vmfb_path=model_dir / "model.vmfb",
                config_path=model_dir / "config.json",
                model_config=model_config,
            )

        processor = ModelProcessor(cache_dir)
        return processor.process_model(model_config)

    def generate(self, config: ModelBatcherConfig) -> List[ModelArtifacts]:
        """
        Prepares a *batch* of model artifacts in a cached directory, in parallel processes.

        Usage:
            @pytest.mark.parametrize("batch_model_artifacts", [[cfg1, cfg2, cfg3]], indirect=True)
            def test_something(batch_model_artifacts):
                a1, a2, a3 = batch_model_artifacts
                ...

        Returns: List[ModelArtifacts] aligned to the order of the provided ModelConfig list.
        """
        model_configs: List[ModelConfig] = config.model_configs
        test_device = config.test_device

        prepared_configs: List[ModelConfig] = []
        processed_artifacts: List[ModelArtifacts | None] = [None] * len(model_configs)

        for idx, model_config in enumerate(model_configs):
            model = self._prepare_model(
                model_config,
                config.irpa_path,
                config.tokenizer_path,
                test_device,
                config.cache_dir,
            )

            if isinstance(model, ModelArtifacts):
                processed_artifacts[idx] = model
            else:
                prepared_configs.append(model)

        if not prepared_configs:
            assert all(artifact is not None for artifact in processed_artifacts)
            return processed_artifacts

        max_workers = len(prepared_configs)
        idx_by_cache_key = {}
        for i, model_config in enumerate(model_configs):
            cache_key = hashlib.md5(str(model_config).encode()).hexdigest()
            idx_by_cache_key[cache_key] = i

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cache_key = {}
            for model_config in prepared_configs:
                cache_key = hashlib.md5(str(model_config).encode()).hexdigest()
                fut = executor.submit(
                    self._process_model, model_config, config.cache_dir
                )
                future_to_cache_key[fut] = cache_key

            for fut in as_completed(future_to_cache_key):
                cache_key = future_to_cache_key[fut]
                idx = idx_by_cache_key[cache_key]
                try:
                    model_artifacts = fut.result()
                except Exception as e:
                    pytest.fail(
                        f"Failed to build artifacts for config {cache_key}: {e!r}"
                    )

                processed_artifacts[idx] = model_artifacts

        assert all(artifact is not None for artifact in processed_artifacts)
        return processed_artifacts
