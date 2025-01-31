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
)
from integration_tests.llm.logging_utils import start_log_group, end_log_group

logger = logging.getLogger(__name__)

MODEL_DIR_CACHE = {}


@pytest.fixture(scope="module")
def pre_process_model(request, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("sglang_benchmark_test")

    logger.info(
        "Preparing model artifacts..." + start_log_group("Preparing model artifacts")
    )

    param_key = hashlib.md5(str(request.param).encode()).hexdigest()
    if (directory := MODEL_DIR_CACHE.get(param_key)) is not None:
        logger.info(
            f"Reusing existing model artifacts directory: {directory}" + end_log_group()
        )
        return MODEL_DIR_CACHE[param_key]

    model_name = request.param["model_name"]
    model_param_file_name = request.param["model_param_file_name"]
    settings = request.param["settings"]
    batch_sizes = request.param["batch_sizes"]

    # Configure model
    config = ModelConfig(
        model_file=model_param_file_name,
        tokenizer_id=model_name,  # Using model_name as tokenizer_id, adjust if needed
        batch_sizes=batch_sizes,
        device_settings=settings,
        source=ModelSource.HUGGINGFACE,
        repo_id=model_name,  # Using model_name as repo_id, adjust if needed
    )

    # Process model through all stages
    processor = ModelProcessor(tmp_dir)
    artifacts = processor.process_model(config)

    logger.info("Model artifacts setup successfully" + end_log_group())
    MODEL_DIR_CACHE[param_key] = tmp_dir
    return tmp_dir


@pytest.fixture(scope="module")
def write_config(request, pre_process_model):
    batch_sizes = request.param["batch_sizes"]
    prefix_sharing_algorithm = request.param["prefix_sharing_algorithm"]

    # Construct the new config filename
    config_path = (
        pre_process_model
        / f"{'_'.join(str(bs) for bs in batch_sizes)}_{prefix_sharing_algorithm}.json"
    )

    # Read the base config file
    base_config_path = pre_process_model / "config.json"
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
