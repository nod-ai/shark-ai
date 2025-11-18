# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import multiprocessing
import os
import pytest
import time
from unittest.mock import patch

pytest.importorskip("sglang")
from sglang import bench_serving

from .utils import (
    SGLangBenchmarkArgs,
    log_jsonl_result,
)

from integration_tests.llm.logging_utils import end_log_group, start_log_group
from integration_tests.llm.server_management import ServerConfig, ServerInstance
from integration_tests.llm.model_management import ModelArtifacts, ModelConfig

logger = logging.getLogger(__name__)

device_settings = {
    "device_flags": [
        "--iree-hal-target-device=hip",
        "--iree-hip-target=gfx942",
    ],
    "device": "hip",
}


@pytest.mark.skip(reason="https://github.com/nod-ai/amdshark-ai/issues/2243")
@pytest.mark.parametrize("request_rate", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize(
    "model_artifacts,server",
    [
        (
            ModelConfig.get(name="llama3.1_8b"),
            {"prefix_sharing": "none"},
        ),
        (
            ModelConfig.get(name="llama3.1_8b"),
            {"prefix_sharing": "trie"},
        ),
    ],
    ids=[
        "llama31_8b_none",
        "llama31_8b_trie",
    ],
    indirect=True,
)
def test_shortfin_benchmark(
    request_rate,
    model_artifacts: ModelArtifacts,
    server,
    request,
):
    # TODO: Remove when multi-device is fixed
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"

    process, port = server

    tmp_dir = model_artifacts.tokenizer_path.parent
    # Run and collect SGLang Serving Benchmark
    benchmark_args = SGLangBenchmarkArgs(
        backend="shortfin",
        num_prompt=10,
        base_url=f"http://localhost:{port}",
        tokenizer=tmp_dir,
        request_rate=request_rate,
    )

    paramid = (
        request.node.callspec.id
    )  # this would be the param id, e.g. llama31_8b_trie

    output_file = (
        tmp_dir
        / f"{benchmark_args.backend}_{benchmark_args.num_prompt}_{benchmark_args.request_rate}_{paramid}.jsonl"
    )
    benchmark_args.output_file = output_file

    logger.info(
        f"Starting benchmark run on {paramid}..."
        + start_log_group(f"Benchmark run on {paramid}")
    )
    logger.info("Running SGLang Benchmark with the following settings:")
    logger.info(f"Test parameterization: {paramid}")
    logger.info(f"Benchmark Args: {benchmark_args}")
    try:
        start = time.time()
        with patch.object(bench_serving, "print", side_effect=logger.info):
            benchmark_process = multiprocessing.Process(
                target=bench_serving.run_benchmark,
                args=(benchmark_args.as_namespace(),),
            )
            benchmark_process.start()
            benchmark_process.join()

        logger.info(f"Benchmark run completed in {str(time.time() - start)} seconds")
        logger.info("\n\n======== RESULTS ========")
        log_jsonl_result(benchmark_args.output_file)
        logger.info("Benchmark run successful" + end_log_group())
    except Exception as e:
        logger.error(e)
        raise e
