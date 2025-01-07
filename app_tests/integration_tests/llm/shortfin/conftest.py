# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import json
import logging
import pytest

pytest.importorskip("transformers")
from ..utils import (
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

MODEL_DIR_CACHE = {}


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

    param_key = hashlib.md5(str(request.param).encode()).hexdigest()
    if (directory := MODEL_DIR_CACHE.get(param_key)) is not None:
        logger.info(
            f"Reusing existing model artifacts directory: {directory}" + end_log_group()
        )
        yield MODEL_DIR_CACHE[param_key]
        return

    repo_id = request.param["repo_id"]
    model_file = request.param["model_file"]
    tokenizer_id = request.param["tokenizer_id"]
    settings = request.param["settings"]
    batch_sizes = request.param["batch_sizes"]
    tmp_dir = tmp_path_factory.mktemp("cpu_llm_server_test")

    # Download model if it doesn't exist
    model_path = tmp_dir / model_file
    download_huggingface_model(tmp_dir, repo_id, model_file)

    # Set up tokenizer if it doesn't exist
    download_tokenizer(tmp_dir, tokenizer_id)

    # Export model
    mlir_path = tmp_dir / "model.mlir"
    config_path = tmp_dir / "config.json"
    export_paged_llm_v1(mlir_path, config_path, model_path, batch_sizes)

    # Compile model
    vmfb_path = tmp_dir / "model.vmfb"
    compile_model(mlir_path, vmfb_path, settings)

    logger.info("Model artifacts setup successfully" + end_log_group())
    MODEL_DIR_CACHE[param_key] = tmp_dir
    yield tmp_dir


@pytest.fixture(scope="module")
def write_config(request, model_test_dir):
    batch_sizes = request.param["batch_sizes"]
    prefix_sharing_algorithm = request.param["prefix_sharing_algorithm"]

    # Construct the new config filename
    config_path = (
        model_test_dir
        / f"{'_'.join(str(bs) for bs in batch_sizes)}_{prefix_sharing_algorithm}.json"
    )

    # Read the base config file
    base_config_path = model_test_dir / "config.json"
    with open(base_config_path, "r") as f:
        config = json.load(f)

    # Override specific fields
    config.update(
        {
            "prefill_batch_sizes": batch_sizes,
            "decode_batch_sizes": batch_sizes,
            "paged_kv_cache": {
                **config.get(
                    "paged_kv_cache", {}
                ),  # Preserve other paged_kv_cache settings
                "prefix_sharing_algorithm": prefix_sharing_algorithm,
            },
        }
    )
    logger.info(f"Saving edited config to: {config_path}\n")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    with open(config_path, "w") as f:
        json.dump(config, f)

    yield config_path


@pytest.fixture(scope="module")
def llm_server(request, model_test_dir, write_config):
    """Start the LLM server.

    Args:
        request (FixtureRequest): The following params are accepted:
            - model_file (str): The model file to download.
            - settings (dict): The settings for starting the server.
        model_test_dir (Tuple[Path, Path]): The paths to the Hugging Face home and the temp dir.

    Yields:
        subprocess.Popen: The server process that was started.
    """
    logger.info("Starting LLM server..." + start_log_group("Starting LLM server"))
    tmp_dir = model_test_dir
    config_path = write_config

    model_file = request.param["model_file"]
    settings = request.param["settings"]

    tokenizer_path = tmp_dir / "tokenizer.json"
    vmfb_path = tmp_dir / "model.vmfb"
    parameters_path = tmp_dir / model_file

    # Start llm server
    server_process, port = start_llm_server(
        tokenizer_path,
        config_path,
        vmfb_path,
        parameters_path,
        settings,
        timeout=60,
    )
    logger.info("LLM server started!" + end_log_group())
    yield server_process, port
    # Teardown: kill the server
    server_process.terminate()
    server_process.wait()
