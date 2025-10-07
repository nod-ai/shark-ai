"""Test fixtures and configurations."""

import hashlib
import os
import pytest
from pathlib import Path
from tokenizers import Tokenizer, Encoding

from ..model_management import (
    ModelProcessor,
    ModelArtifacts,
    ModelConfig,
    ModelSource,
)
from ..server_management import ServerInstance, ServerConfig

from ..device_settings import get_device_settings_by_name


def pytest_addoption(parser):
    parser.addoption(
        "--test_device",
        action="store",
        metavar="NAME",
        default=None,  # you must specify a device to test on
        help="Select device name to compile models to and run tests on ('cpu', 'gfx90a', 'gfx942', ...); see app_tests/integration_tests/llm/device_settings.py for full list of options.",
    )


@pytest.fixture(scope="session")
def test_device(request):
    ret = request.config.option.test_device
    if ret is None:
        raise ValueError("--test_device not specified")
    return ret


@pytest.fixture(scope="session")
def irpa_path():
    path = os.environ.get("IRPA_PATH")
    return path


@pytest.fixture(scope="session")
def tokenizer_path():
    path = os.environ.get("TOKENIZER_PATH")
    return path


@pytest.fixture(scope="session")
def model_artifacts(tmp_path_factory, request, test_device, irpa_path, tokenizer_path):
    """Prepares model artifacts in a cached directory."""
    model_config: ModelConfig = request.param
    settings_key = test_device

    if test_device == "cpu" and model_config.tensor_parallelism_size is not None:
        pytest.skip("Skipping CPU tests with tensor parallelism")

    if test_device == "cpu" and model_config.has_prefill_position is not None:
        pytest.skip(
            reason="Skipping CPU tests with prefill position due to compilation error"
        )

    if model_config.source == ModelSource.LOCAL:
        if irpa_path is None:
            pytest.fail("IRPA path must be specified for LOCAL models")
        if tokenizer_path is None:
            pytest.fail("Tokenizer path must be specified for LOCAL models")

        model_config.irpa_path = Path(irpa_path)
        model_config.tokenizer_path = Path(tokenizer_path)

    if (
        model_config.tensor_parallelism_size is not None
        and model_config.tensor_parallelism_size > 1
    ):
        settings_key += f"_tp{model_config.tensor_parallelism_size}"
    model_config.device_settings = get_device_settings_by_name(settings_key)
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
            model_config=model_config,
        )

    # Process model and create artifacts
    processor = ModelProcessor(cache_dir)
    return processor.process_model(model_config)


@pytest.fixture(scope="module")
def server(model_artifacts, request):
    """Starts and manages the test server."""
    model_config = model_artifacts.model_config

    server_config = ServerConfig(
        artifacts=model_artifacts,
        device_settings=model_config.device_settings,
        prefix_sharing_algorithm=request.param.get("prefix_sharing", "none"),
        num_beams=request.param.get("num_beams", 1),
        chunk_block_size=request.param.get("chunk_block_size", None),
    )

    server_instance = ServerInstance(server_config)
    server_instance.start()
    process, port, config = (
        server_instance.process,
        server_instance.port,
        server_instance.config,
    )
    yield process, port, config

    # Teardown, if process is still running
    if process.poll() is None:
        process.terminate()
        process.wait()


@pytest.fixture(scope="module")
def generate_service(model_artifacts, request):
    """Starts and manages the test server."""
    model_config = model_artifacts.model_config

    server_config = ServerConfig(
        artifacts=model_artifacts,
        device_settings=model_config.device_settings,
        prefix_sharing_algorithm=request.param.get("prefix_sharing", "none"),
    )

    server_instance = ServerInstance(server_config)
    server_instance.port = 0
    with server_instance.start_service_only() as gs:
        yield gs


@pytest.fixture(scope="module")
def encoded_prompt(model_artifacts: ModelArtifacts, request) -> list[int]:
    tokenizer = Tokenizer.from_file(str(model_artifacts.tokenizer_path))
    return tokenizer.encode(request.param).ids
