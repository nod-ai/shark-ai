# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import time
from unittest.mock import patch

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


@pytest.mark.parametrize(
    "request_rate,tokenizer_id",
    [(req_rate, "NousResearch/Meta-Llama-3-8B") for req_rate in [1, 2, 4, 8, 16, 32]],
)
def test_sglang_benchmark(request_rate, tokenizer_id, sglang_args, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("sglang_benchmark_test")

    # Download tokenizer using ModelProcessor
    config = ModelConfig(
        model_file="tokenizer.json",  # Only need tokenizer
        tokenizer_id=tokenizer_id,
        batch_sizes=(1,),  # Not relevant for tokenizer only
        device_settings=None,  # Not relevant for tokenizer only
        source=ModelSource.HUGGINGFACE,
        repo_id=tokenizer_id,
    )
    processor = ModelProcessor(tmp_dir)
    artifacts = processor.process_model(config)

    logger.info("Beginning SGLang benchmark test...")

    port = sglang_args
    base_url = f"http://localhost:{port}"

    # Wait for server using ServerInstance's method
    server = ServerInstance(
        None
    )  # We don't need config since we're just using wait_for_ready
    server.port = int(port)  # Set port manually since we didn't start the server
    server.wait_for_ready(
        timeout=600
    )  # High timeout for model artifacts download and server startup

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
