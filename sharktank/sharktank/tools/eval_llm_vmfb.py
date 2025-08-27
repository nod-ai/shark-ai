# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json

from sharktank.utils.tokenizer import load_tokenizer
from sharktank.models.llm.config import ServiceConfig
from sharktank.utils.llm_utils import IreeInstance, LlmInstance, LlmPerplexityEval


def main(dataset, vmfb, config, irpa, tokenizer, expected_err):
    tokenizer = load_tokenizer(tokenizer)
    iree = IreeInstance(devices=["hip://0"], vmfb=vmfb, parameters=irpa)
    server_config = ServiceConfig.load(config)
    llm = LlmInstance.load(iree, server_config)

    runner = llm.make_perplexity_eval()

    with open(dataset, "r") as dataset:
        dataset = LlmPerplexityEval.Dataset(**json.load(dataset))

    results = runner.run_dataset(dataset=dataset, tokenizer=tokenizer)
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
    parser.add_argument("--vmfb", help="vmfb file path", required=True)
    parser.add_argument("--config", help="json config file for server", required=True)
    parser.add_argument("--tokenizer", help="json tokenizer config file", required=True)
    parser.add_argument(
        "--expected-err", help="expected error in the difference", type=float
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        irpa=args.irpa,
        vmfb=args.vmfb,
        config=args.config,
        tokenizer=args.tokenizer,
        expected_err=args.expected_err,
    )
