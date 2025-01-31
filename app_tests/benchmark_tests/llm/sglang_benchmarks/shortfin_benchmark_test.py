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
from integration_tests.llm.model_management import ModelArtifacts

logger = logging.getLogger(__name__)

device_settings = {
    "device_flags": [
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx942",
    ],
    "device": "hip",
}


@pytest.mark.parametrize(
    "request_rate,model_param_file_name",
    [
        (req_rate, "meta-llama-3.1-8b-instruct.f16.gguf")
        for req_rate in [1, 2, 4, 8, 16, 32]
    ],
)
@pytest.mark.parametrize(
    "pre_process_model,write_config",
    [
        pytest.param(
            {
                "model_name": "llama3_8B_fp16",
                "model_param_file_name": "meta-llama-3.1-8b-instruct.f16.gguf",
                "settings": device_settings,
                "batch_sizes": [1, 4],
            },
            {"batch_sizes": [1, 4], "prefix_sharing_algorithm": "none"},
        ),
        pytest.param(
            {
                "model_name": "llama3_8B_fp16",
                "model_param_file_name": "meta-llama-3.1-8b-instruct.f16.gguf",
                "settings": device_settings,
                "batch_sizes": [1, 4],
            },
            {"batch_sizes": [1, 4], "prefix_sharing_algorithm": "trie"},
        ),
    ],
    indirect=True,
)
def test_shortfin_benchmark(
    request_rate, model_param_file_name, pre_process_model, write_config
):
    # TODO: Remove when multi-device is fixed
    os.environ["ROCR_VISIBLE_DEVICES"] = "1"

    tmp_dir = pre_process_model

    config_path = write_config
    prefix_sharing_algorithm = config_path.stem.split("_")[-1]
    vmfb_path = tmp_dir / "model.vmfb"
    tokenizer_path = tmp_dir / "tokenizer.json"
    model_path = tmp_dir / model_param_file_name

    # Start shortfin llm server
    server_config = ServerConfig(
        artifacts=ModelArtifacts(
            weights_path=model_path,
            tokenizer_path=tokenizer_path,
            mlir_path=tmp_dir / "model.mlir",
            vmfb_path=vmfb_path,
            config_path=config_path,
        ),
        device_settings=device_settings,
    )
    server = ServerInstance(server_config)
    server.start()

    # Run and collect SGLang Serving Benchmark
    benchmark_args = SGLangBenchmarkArgs(
        backend="shortfin",
        num_prompt=10,
        base_url=f"http://localhost:{server.port}",
        tokenizer=tmp_dir,
        request_rate=request_rate,
    )
    output_file = (
        tmp_dir
        / f"{benchmark_args.backend}_{benchmark_args.num_prompt}_{benchmark_args.request_rate}_{prefix_sharing_algorithm}.jsonl"
    )
    benchmark_args.output_file = output_file

    logger.info(
        f"Starting benchmark run with prefix sharing algorith {prefix_sharing_algorithm}..."
        + start_log_group(f"Benchmark run with {prefix_sharing_algorithm} algorithm")
    )
    logger.info("Running SGLang Benchmark with the following settings:")
    logger.info(f"Prefix sharing algorith: {prefix_sharing_algorithm}")
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

    server.stop()
