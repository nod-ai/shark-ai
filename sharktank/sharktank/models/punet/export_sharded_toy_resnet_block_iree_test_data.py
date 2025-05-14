# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from argparse import ArgumentParser
from typing import Optional
from pathlib import Path
import torch

from .testing import export_sharded_toy_resnet_block_iree_test_data


def main(args: Optional[list[str]] = None):
    parser = ArgumentParser(
        description=(
            "Export test data for toy-sized Resnet block."
            " This includes the program MLIR, parameters, sample input and expected output."
            " Exports a float32 model variant."
            " The expected output is in float64 precision."
        )
    )
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args(args=args)
    output_dir = Path(args.output_dir)
    export_sharded_toy_resnet_block_iree_test_data(
        mlir_path=output_dir / "model.mlir",
        parameters_path=output_dir / "model.irpa",
        input_args_path=output_dir / "input_args.irpa",
        expected_results_path=output_dir / "expected_results.irpa",
        target_dtype=torch.float32,
    )


if __name__ == "__main__":
    main()
