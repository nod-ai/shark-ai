# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import time
from unittest.mock import patch
from transformers import AutoTokenizer
import os
import requests

pytest.importorskip("sglang")
from sglang import bench_serving

from .utils import SGLangBenchmarkArgs, log_jsonl_result
from integration_tests.llm.model_management import (
    ModelConfig,
    ModelProcessor,
    ModelSource,
)
from integration_tests.llm.server_management import ServerInstance

logger = logging.getLogger(__name__)


def download_tokenizer(local_dir, tokenizer_id):
    # Set up tokenizer if it doesn't exist
    tokenizer_path = local_dir / "tokenizer.json"
    logger.info(f"Preparing tokenizer_path: {tokenizer_path}...")
    if not os.path.exists(tokenizer_path):
        logger.info(f"Downloading tokenizer {tokenizer_id} from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
        )
        tokenizer.save_pretrained(local_dir)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
    else:
        logger.info("Using cached tokenizer")


def wait_for_server(url, timeout):
    logger.info(f"Waiting for server to start at {url}...")
    start = time.time()
    elapsed = 0
    while elapsed <= timeout:
        try:
            requests.get(f"{url}/health")
            logger.info("Server successfully started")
            return
        except requests.exceptions.ConnectionError:
            logger.info(
                f"Server has not started yet; waited {elapsed} seconds; timeout: {timeout} seconds."
            )
            time.sleep(1)
        elapsed = time.time() - start
    raise TimeoutError(f"Server did not start within {timeout} seconds at {url}")


@pytest.mark.skip(reason="https://github.com/nod-ai/amdshark-ai/issues/2243")
@pytest.mark.parametrize(
    "request_rate,tokenizer_id",
    [(req_rate, "NousResearch/Meta-Llama-3-8B") for req_rate in [1, 2, 4, 8, 16, 32]],
)
def test_sglang_benchmark(request_rate, tokenizer_id, sglang_args, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("sglang_benchmark_test")

    download_tokenizer(tmp_dir, tokenizer_id)

    logger.info("Beginning SGLang benchmark test...")

    port = sglang_args
    base_url = f"http://localhost:{port}"

    # Setting a high timeout gives enough time for downloading model artifacts
    # and starting up server... Takes a little longer than shortfin.
    wait_for_server(base_url, timeout=600)

    benchmark_args = SGLangBenchmarkArgs(
        backend="sglang",
        num_prompt=10,
        base_url=f"http://localhost:{port}",
        tokenizer=tmp_dir,
        request_rate=request_rate,
    )
    output_file = (
        tmp_dir
        / f"{benchmark_args.backend}_{benchmark_args.num_prompt}_{benchmark_args.request_rate}.jsonl"
    )
    benchmark_args.output_file = output_file

    logger.info("Running SGLang Benchmark with the following args:")
    logger.info(benchmark_args)

    try:
        start = time.time()
        with patch.object(bench_serving, "print", side_effect=logger.info):
            bench_serving.run_benchmark(
                benchmark_args.as_namespace(),
            )
        logger.info(f"Benchmark run completed in {str(time.time() - start)} seconds")
        logger.info("======== RESULTS ========")
        log_jsonl_result(benchmark_args.output_file)
    except Exception as e:
        logger.error(e)
        raise e
