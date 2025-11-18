# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import iree.compiler
import json
import logging

from amdsharktank.examples.export_paged_llm_v1 import (
    export_llm_v1,
    ExportConfig,
    LlamaHParams,
    ParallelismConfig,
    LlamaModelConfig,
)
from amdsharktank.models.llm.config import ServiceConfig
from amdsharktank.types import Dataset
from amdsharktank.utils.llm_utils import IreeInstance, LlmInstance, LlmPerplexityEval
from amdsharktank.utils.tokenizer import load_tokenizer
from amdsharktank.types.pipelining import pipeline_parallelize_llm_theta


def export_ir(irpa: str, pipeline_parallelism_size: int):
    logging.log(logging.INFO, "Exporting IR")
    dataset = Dataset.load(irpa, file_type="irpa")
    dataset.root_theta
    dataset.properties

    parallelism_config = ParallelismConfig.default_config(
        block_count=len(dataset.root_theta.tensor("blk")), pp=pipeline_parallelism_size
    )
    pipeline_parallelize_llm_theta(dataset.root_theta, parallelism_config)

    llama_config = LlamaModelConfig.from_dataset(
        dataset=dataset, block_seq_stride=32, parallelism_config=parallelism_config
    )

    # Configure model export config from cli args:
    export_config = ExportConfig(
        device_block_count=4096,
        bs_prefill=[4],
        bs_decode=[32],
        logits_normalization="none",
    )

    ir, config = export_llm_v1(
        llama_config=llama_config, theta=dataset.root_theta, export_config=export_config
    )
    ir = ir.mlir_module.get_asm()
    return ir, config


def compile_ir(ir, iree_hal_target_devices: list[str], iree_hip_target: str):
    logging.log(
        logging.INFO, f"Compiling VMFB on {iree_hal_target_devices} - {iree_hip_target}"
    )
    extra_args = [
        f"--iree-hal-target-device={device}" for device in iree_hal_target_devices
    ]
    extra_args += [f"--iree-hip-target={iree_hip_target}"]
    vmfb = iree.compiler.compile_str(ir, extra_args=extra_args)
    return vmfb


def get_instance(
    vmfb: str | None,
    config: str | None,
    irpa: str,
    iree_hal_target_devices: list[str] | None,
    iree_hip_target: str | None,
    pipeline_parallelism_size: int,
) -> LlmInstance:
    if vmfb is None:
        if iree_hal_target_devices is None:
            raise ValueError("--iree-hal-target-device is required")

        if iree_hip_target is None:
            raise ValueError("--iree-hip-target is required")

        if config is not None:
            raise ValueError("Config found without corresponding vmfb")
        ir, config = export_ir(irpa, pipeline_parallelism_size)
        vmfb = compile_ir(ir, iree_hal_target_devices, iree_hip_target)

    if isinstance(config, str):
        config: ServiceConfig = ServiceConfig.load(config)

    devices = [f"hip://{i}" for i in range(pipeline_parallelism_size)]
    iree = IreeInstance(devices=devices, vmfb=vmfb, parameters=irpa)
    llm = LlmInstance.load(iree, config)
    return llm


def main(
    dataset: str,
    vmfb: str | None,
    config: str | None,
    irpa: str,
    tokenizer: str,
    min_context: int,
    expected_err: float | None,
    iree_hal_target_devices: list[str] | None,
    iree_hip_target: str | None,
    pipeline_parallelism_size: int,
):
    tokenizer = load_tokenizer(tokenizer)
    llm = get_instance(
        vmfb=vmfb,
        config=config,
        irpa=irpa,
        iree_hal_target_devices=iree_hal_target_devices,
        iree_hip_target=iree_hip_target,
        pipeline_parallelism_size=pipeline_parallelism_size,
    )
    runner = llm.make_perplexity_eval()

    with open(dataset, "r") as dataset:
        dataset = LlmPerplexityEval.Dataset(**json.load(dataset))

    results = runner.run_dataset(
        dataset=dataset, tokenizer=tokenizer, min_context=min_context
    )
    print(json.dumps(results.as_dict(), indent=1))

    if expected_err:
        if not all([str(id) in dataset.scores for id in dataset.ids]):
            raise ValueError("Not all baselines available in dataset")

        err = dataset.compare(results)
        if err > expected_err:
            raise ValueError(f"Exceeded allowable error ({expected_err}, found {err})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to dataset", required=True)
    parser.add_argument("--irpa", help="IRPA parameters file", required=True)
    parser.add_argument("--vmfb", help="vmfb file path")
    parser.add_argument("--config", help="json config file for server")
    parser.add_argument(
        "--tokenizer", help="json tokenizer config folder", required=True
    )
    parser.add_argument(
        "--expected-err", help="expected error in the difference", type=float
    )
    parser.add_argument(
        "--min-context", help="required context length", type=int, default=0
    )
    # iree-hal-target-device needs to support multiple devices
    parser.add_argument(
        "--iree-hal-target-device",
        help="Target device(s) if compiling",
        action="append",
    )
    parser.add_argument("--iree-hip-target", help="Iree hip target")
    parser.add_argument(
        "--pipeline-parallelism-size",
        help="Pipeline parallelism size",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    # TODO: This is deceiving, it's the tokenizer directory
    # if not a directory, raise an error
    if not os.path.isdir(args.tokenizer):
        raise ValueError(
            "Provide the path to the tokenizer's folder rather than the json itself."
        )

    main(
        dataset=args.dataset,
        irpa=args.irpa,
        vmfb=args.vmfb,
        config=args.config,
        tokenizer=args.tokenizer,
        min_context=args.min_context,
        expected_err=args.expected_err,
        iree_hal_target_devices=args.iree_hal_target_device,
        iree_hip_target=args.iree_hip_target,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
    )
