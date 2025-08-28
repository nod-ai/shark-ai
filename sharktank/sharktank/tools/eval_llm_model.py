# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import math
import torch

from sharktank.utils.tokenizer import load_tokenizer
from sharktank.utils.llm_utils import (
    TorchInstance,
    LlmInstance,
    llama_config_page_size,
    LlmPerplexityEval,
)


def main(device, dataset, irpa, tokenizer, min_context, expected_err):
    torch.set_default_device(device)
    tokenizer = load_tokenizer(tokenizer)
    torch_instance = TorchInstance.load(irpa, device=device)

    page_size = llama_config_page_size(torch_instance.config)
    block_count = 512

    llm = LlmInstance(
        model_instance=torch_instance,
        page_size=page_size,
        block_seq_stride=torch_instance.config.block_seq_stride,
        block_count=block_count,
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
    parser.add_argument(
        "--device", help="Torch device to use for computation", default="cuda"
    )
    parser.add_argument("--dataset", help="Path to dataset", required=True)
    parser.add_argument("--irpa", help="IRPA parameters file", required=True)
    parser.add_argument("--tokenizer", help="json tokenizer config file", required=True)
    parser.add_argument(
        "--expected-err", help="expected error in the difference", type=float
    )
    parser.add_argument(
        "--min-context", help="required context length", type=int, default=0
    )
    args = parser.parse_args()
    main(
        device=args.device,
        dataset=args.dataset,
        irpa=args.irpa,
        tokenizer=args.tokenizer,
        min_context=args.min_context,
        expected_err=args.expected_err,
    )
