# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import functools
import torch

from amdsharktank.transforms.dataset import convert_dtype
from amdsharktank.types import serialized_name_to_dtype
from amdsharktank.utils import cli


def main(args: list[str] | None = None):
    parser = cli.create_parser(
        description="Convert a dataset (model weights). For example convert the dtype."
    )
    cli.add_input_dataset_options(parser)
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "--dtype",
        action="append",
        type=str,
        default=[],
        help='Convert all tensors with dtype form one to another. E.g. "--dtype=float16->bfloat16"',
    )
    args = cli.parse(parser, args=args)

    dataset = cli.get_input_dataset(args)
    dtype_conversion_map = _construct_dtype_map(args)
    dataset.transform(functools.partial(convert_dtype, dtype_map=dtype_conversion_map))
    dataset.save(args.output_irpa_file)


def _construct_dtype_map(args: argparse.Namespace) -> dict[torch.dtype, torch.dtype]:
    res = {}
    for map_as_str in args.dtype:
        parts = map_as_str.split("->")
        assert (
            len(parts) == 2
        ), "The map must be of the form source_dtype->target_dtype. E.g. float16->bfloat16"
        src_dtype = serialized_name_to_dtype(parts[0])
        tgt_dtype = serialized_name_to_dtype(parts[1])
        res[src_dtype] = tgt_dtype
    return res


if __name__ == "__main__":
    main()
