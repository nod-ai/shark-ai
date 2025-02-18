# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import pytest
from pathlib import Path

from ..model_management import (
    ModelProcessor,
    ModelConfig,
    ModelSource,
)
from ..server_management import ServerInstance, ServerConfig

pytest.importorskip("sglang")
import sglang as sgl
from sglang.lang.chat_template import get_chat_template

pytest.importorskip("sentence_transformers")
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def register_shortfin_backend(port):
    backend = sgl.Shortfin(
        chat_template=get_chat_template("llama-3-instruct"),
        base_url=f"http://localhost:{port}",
    )
    sgl.set_default_backend(backend)


@pytest.fixture(scope="module")
def model_artifacts(request, tmp_path_factory):
    device_settings = request.param["device_settings"]
    tmp_dir = tmp_path_factory.mktemp("sglang_integration_tests")

    model_config = ModelConfig(
        source=ModelSource.HUGGINGFACE_FROM_GGUF,
        repo_id="SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
        model_file="meta-llama-3.1-8b-instruct.f16.gguf",
        tokenizer_id="NousResearch/Meta-Llama-3.1-8B",
        batch_sizes=(1, 4),
        device_settings=device_settings,
    )

    processor = ModelProcessor(tmp_dir)
    return processor.process_model(model_config)


@pytest.fixture(scope="module")
def start_server(request, model_artifacts):
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    device_settings = request.param["device_settings"]

    server_config = ServerConfig(
        artifacts=model_artifacts,
        device_settings=device_settings,
        prefix_sharing_algorithm="none",
    )

    server_instance = ServerInstance(server_config)
    server_instance.start()
    process, port = server_instance.process, server_instance.port

    yield process, port

    process.terminate()
    process.wait()


@pytest.fixture(scope="module")
def load_comparison_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model
