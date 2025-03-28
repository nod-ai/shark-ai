# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import json
import logging
import os
import pytest
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from integration_tests.llm.model_management import (
    ModelConfig,
    ModelProcessor,
    ModelSource,
    ModelArtifacts,
)
from integration_tests.llm.server_management import ServerInstance, ServerConfig

from integration_tests.llm import device_settings
from integration_tests.llm.logging_utils import start_log_group, end_log_group

logger = logging.getLogger(__name__)

MODEL_DIR_CACHE = {}


# we can replace this with an import after #890 merges
TEST_MODELS = {
    "llama3.1_8b": ModelConfig(
        source=ModelSource.HUGGINGFACE_FROM_GGUF,
        repo_id="SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
        model_file="meta-llama-3.1-8b-instruct.f16.gguf",
        tokenizer_id="NousResearch/Meta-Llama-3.1-8B",
        batch_sizes=(1, 4),
        device_settings=device_settings.GFX942,
    ),
}


@pytest.fixture(scope="module")
def model_artifacts(tmp_path_factory, request):
    """Prepares model artifacts in a cached directory."""
    model_config = request.param
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
    model_config = model_artifacts.model_config

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


def pytest_addoption(parser):
    parser.addoption(
        "--port",
        action="store",
        default="30000",
        help="Port that SGLang server is running on",
    )


@pytest.fixture(scope="module")
def sglang_args(request):
    return request.config.getoption("--port")
