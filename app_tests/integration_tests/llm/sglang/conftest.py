# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import logging
import os
import pytest

from ..utils import (
    find_available_port,
    start_llm_server,
    download_with_hf_datasets,
    export_paged_llm_v1,
    compile_model,
)

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
def pre_process_model(request, tmp_path_factory):
    device_settings = request.param["device_settings"]
    tmp_dir = tmp_path_factory.mktemp("sglang_integration_tests")

    # Download model
    model_params_path = tmp_dir / "meta-llama-3.1-8b-instruct.f16.gguf"
    download_with_hf_datasets(tmp_dir, "llama3_8B_fp16")

    # Export to mlir
    mlir_path = tmp_dir / "model.mlir"
    config_path = tmp_dir / "config.json"
    batch_sizes = [1, 4]
    export_paged_llm_v1(
        mlir_path,
        config_path,
        model_params_path,
        batch_sizes,
    )

    # Compile Model
    vmfb_path = tmp_dir / "model.vmfb"
    compile_model(
        mlir_path,
        vmfb_path,
        device_settings,
    )

    return tmp_dir


@pytest.fixture(scope="module")
def start_server(request, pre_process_model):
    os.environ["ROCR_VISIBLE_DEVICES"] = "1"
    device_settings = request.param["device_settings"]

    export_dir = pre_process_model

    tokenizer_path = export_dir / "tokenizer.json"
    model_params_path = export_dir / "meta-llama-3.1-8b-instruct.f16.gguf"
    vmfb_path = export_dir / "model.vmfb"
    config_path = export_dir / "config.json"

    logger.info("Starting server...")
    server_process, port = start_llm_server(
        tokenizer_path,
        config_path,
        vmfb_path,
        model_params_path,
        device_settings,
        timeout=30,
    )
    logger.info("Server started")

    yield server_process, port

    server_process.terminate()
    server_process.wait()


@pytest.fixture(scope="module")
def load_comparison_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model
