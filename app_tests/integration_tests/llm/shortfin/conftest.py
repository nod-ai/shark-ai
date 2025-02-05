"""Test fixtures and configurations."""

import hashlib
import pytest
from pathlib import Path
from tokenizers import Tokenizer, Encoding

from ..model_management import (
    ModelProcessor,
    ModelConfig,
    ModelSource,
    AzureConfig,
    ModelArtifacts,
)
from ..server_management import ServerInstance, ServerConfig
from .. import device_settings

# Example model configurations
TEST_MODELS = {
    "open_llama_3b": ModelConfig(
        source=ModelSource.HUGGINGFACE,
        repo_id="SlyEcho/open_llama_3b_v2_gguf",
        model_file="open-llama-3b-v2-f16.gguf",
        tokenizer_id="openlm-research/open_llama_3b_v2",
        batch_sizes=(1, 4),
        device_settings=device_settings.CPU,
    ),
    "llama3.1_8b": ModelConfig(
        source=ModelSource.HUGGINGFACE,
        repo_id="SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
        model_file="meta-llama-3.1-8b-instruct.f16.gguf",
        tokenizer_id="NousResearch/Meta-Llama-3.1-8B",
        batch_sizes=(1, 4),
        device_settings=device_settings.CPU,
    ),
    "azure_llama": ModelConfig(
        source=ModelSource.AZURE,
        azure_config=AzureConfig(
            account_name="sharkblobs",
            container_name="halo-models",
            blob_path="llm-dev/llama3_8b/8b_f16.irpa",
        ),
        model_file="azure-llama.irpa",
        tokenizer_id="openlm-research/open_llama_3b_v2",
        batch_sizes=(1, 4),
        device_settings=device_settings.CPU,
    ),
}


@pytest.fixture(scope="module")
def model_artifacts(tmp_path_factory, request):
    """Prepares model artifacts in a cached directory."""
    model_config = TEST_MODELS[request.param]
    cache_key = hashlib.md5(str(model_config).encode()).hexdigest()

    cache_dir = tmp_path_factory.mktemp("model_cache")
    model_dir = cache_dir / cache_key

    # Return cached artifacts if available
    if model_dir.exists():
        return ModelArtifacts(
            weights_path=model_dir / model_config.model_file,
            tokenizer_path=model_dir / "tokenizer.json",
            mlir_path=model_dir / "model.mlir",
            vmfb_path=model_dir / "model.vmfb",
            config_path=model_dir / "config.json",
        )

    # Process model and create artifacts
    processor = ModelProcessor(cache_dir)
    return processor.process_model(model_config)


@pytest.fixture(scope="module")
def server(model_artifacts, request):
    """Starts and manages the test server."""
    model_id = request.param["model"]
    model_config = TEST_MODELS[model_id]

    server_config = ServerConfig(
        artifacts=model_artifacts,
        device_settings=model_config.device_settings,
        prefix_sharing_algorithm=request.param.get("prefix_sharing", "none"),
    )

    server_instance = ServerInstance(server_config)
    server_instance.start()
    process, port = server_instance.process, server_instance.port
    yield process, port

    process.terminate()
    process.wait()


@pytest.fixture(scope="module")
def encoded_prompt(model_artifacts: ModelArtifacts, request) -> list[int]:
    tokenizer = Tokenizer.from_file(str(model_artifacts.tokenizer_path))
    return tokenizer.encode(request.param).ids
