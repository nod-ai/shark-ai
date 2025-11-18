# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from amdsharktank.types.theta import Dataset
from amdsharktank.models.llm import *
from amdsharktank.layers.configs import *

import argparse
import math
import numpy
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-path", type=str, required=True)
    parser.add_argument("--irpa-path", type=str, required=True)
    parser.add_argument("--results-path", type=str)

    args = parser.parse_args()

    dataset = Dataset.load(args.irpa_path)
    properties = dataset.properties
    theta = dataset.root_theta

    config = LlamaModelConfig(
        hp=LlamaHParams(**properties["hparams"]),
        block_seq_stride=8,
        activation_dtype=torch.float32,
        attention_dtype=torch.float32,
    )

    model = PagedLlmModelV1(theta=theta, config=config)
    ids = torch.from_numpy(numpy.load(args.ids_path))
    results = model.prefill(tokens=ids)

    if args.results_path:
        expected = torch.from_numpy(numpy.load(args.results_path))
        diff = expected - results
        expected_ce = torch.nn.functional.cross_entropy(expected[0, :-1], ids[0, 1:])
        result_ce = torch.nn.functional.cross_entropy(results[0, :-1], ids[0, 1:])
        sqdiff = math.sqrt(torch.sum(diff * diff) / diff.numel())
        print(f"Squared error {sqdiff}")
        print(f"Expected Cross Entropy {expected_ce}")
        print(f"Result Cross Entropy {result_ce}")
