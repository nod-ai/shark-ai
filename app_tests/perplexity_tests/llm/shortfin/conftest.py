# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import logging
import os
from pathlib import Path
import pytest
import shutil

pytest.importorskip("transformers")
from integration_tests.llm.utils import (
    download_huggingface_model,
    download_tokenizer,
    export_paged_llm_v1,
    compile_model,
    find_available_port,
    start_llm_server,
    start_log_group,
    end_log_group,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_test_dir(request, tmp_path_factory):
    """Prepare model artifacts for starting the LLM server.

    Args:
        request (FixtureRequest): The following params are accepted:
            - repo_id (str): The Hugging Face repo ID.
            - model_file (str): The model file to download.
            - tokenizer_id (str): The tokenizer ID to download.
            - settings (dict): The settings for sharktank export.
            - batch_sizes (list): The batch sizes to use for the model.
        tmp_path_factory (TempPathFactory): Temp dir to save artifacts to.

    Yields:
        Tuple[Path, Path]: The paths to the Hugging Face home and the temp dir.
    """
    logger.info(
        "Preparing model artifacts..." + start_log_group("Preparing model artifacts")
    )

    repo_id = request.param["repo_id"]
    model_file = request.param["model_file"]
    tokenizer_id = request.param["tokenizer_id"]
    settings = request.param["settings"]
    batch_sizes = request.param["batch_sizes"]
    prefix_sharing_algorithm = request.param["prefix_sharing_algorithm"]

    tmp_dir = tmp_path_factory.mktemp("cpu_llm_server_test")
    hf_home = os.environ.get("HF_HOME", None)
    hf_home = Path(hf_home) if hf_home is not None else tmp_dir
    try:
        # Download model if it doesn't exist
        model_path = hf_home / model_file
        download_huggingface_model(hf_home, repo_id, model_file)

        # Set up tokenizer if it doesn't exist
        download_tokenizer(hf_home, tokenizer_id)

        # Export model
        mlir_path = tmp_dir / "model.mlir"
        config_path = tmp_dir / "config.json"
        export_paged_llm_v1(mlir_path, config_path, model_path, batch_sizes)

        # Compile model
        vmfb_path = tmp_dir / "model.vmfb"
        compile_model(mlir_path, vmfb_path, settings)

        # Write config
        edited_config_path = tmp_dir / "edited_config.json"
        config = {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": 2048,
            "attn_head_count": 32,
            "attn_head_dim": 100,
            "prefill_batch_sizes": batch_sizes,
            "decode_batch_sizes": batch_sizes,
            "transformer_block_count": 26,
            "paged_kv_cache": {
                "block_seq_stride": 16,
                "device_block_count": 256,
                "prefix_sharing_algorithm": prefix_sharing_algorithm,
            },
        }
        logger.info(f"Saving edited config to: {edited_config_path}\n")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        with open(edited_config_path, "w") as f:
            json.dump(config, f)
        logger.info("Model artifacts setup successfully" + end_log_group())
        yield hf_home, tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


@pytest.fixture(scope="module")
def available_port():
    return find_available_port()


@pytest.fixture(scope="module")
def llm_server(request, model_test_dir, available_port):
    """Start the LLM server.

    Args:
        request (FixtureRequest): The following params are accepted:
            - model_file (str): The model file to download.
            - settings (dict): The settings for starting the server.
        model_test_dir (Tuple[Path, Path]): The paths to the Hugging Face home and the temp dir.
        available_port (int): The available port to start the server on.

    Yields:
        subprocess.Popen: The server process that was started.
    """
    logger.info("Starting LLM server..." + start_log_group("Starting LLM server"))
    hf_home, tmp_dir = model_test_dir
    model_file = request.param["model_file"]
    settings = request.param["settings"]

    tokenizer_path = hf_home / "tokenizer.json"
    config_path = tmp_dir / "edited_config.json"
    vmfb_path = tmp_dir / "model.vmfb"
    parameters_path = hf_home / model_file

    # Start llm server
    server_process = start_llm_server(
        available_port,
        tokenizer_path,
        config_path,
        vmfb_path,
        parameters_path,
        settings,
    )
    logger.info("LLM server started!" + end_log_group())
    yield server_process
    # Teardown: kill the server
    server_process.terminate()
    server_process.wait()
