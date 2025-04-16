# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..testing import make_random_theta_from_dataset

from sharktank.layers.configs import LlamaHParams
from sharktank.models.llama.llama import LlamaModelConfig
from sharktank.types import Dataset

import argparse
import torch
from ....types.tensors import *
from ....utils import cli

"""Generates a randomized irpa file using a normal distribution given
a llama irpa file in order to maintain the same mean and standard
deviation as the original weights.

Usage:
  $ python -m sharktank.models.llama.tools.randomize_llama_weights \
    --irpa-file=llama3.1_8b_f16.irpa \
    --output=random_llama3.1_8b_f16.irpa
"""

parser = cli.create_parser()
cli.add_input_dataset_options(parser)
cli.add_model_options(parser)
parser.add_argument("-s", "--seed", default=12345)
parser.add_argument("-o", "--output", default="/tmp/random_llama.irpa")
args = cli.parse(parser)


def generate(seed):
    torch.manual_seed(seed)
    dtype = torch.float16

    device = torch.device(args.device) if args.device else None
    dataset = cli.get_input_dataset(args)
    config = LlamaModelConfig(
        hp=LlamaHParams.from_gguf_props(dataset.properties),
        block_seq_stride=args.block_seq_stride,
        device=device,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        attention_kernel=args.attention_kernel,
        kv_cache_dtype=args.kv_cache_dtype,
        use_hf=args.use_hf,
        tensor_parallelism_size=args.tensor_parallelism_size,
    )

    theta = make_random_theta_from_dataset(dataset)
    return theta, config


def main():
    args = parser.parse_args()
    theta, config = generate(args.seed)

    config_dict = config.hp.to_gguf_props()

    dataset = Dataset(config_dict, theta)
    dataset.save(args.output)


if __name__ == "__main__":
    main()
