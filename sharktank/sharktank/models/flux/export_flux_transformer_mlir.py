# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from argparse import ArgumentParser
from typing import Optional
from pathlib import Path

from .export import (
    flux_transformer_default_batch_sizes,
    export_flux_transformer_model_mlir,
)


def main(args: Optional[list[str]] = None):
    parser = ArgumentParser(
        description="Export Flux transformer MLIR from a parameters file."
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--parameters-path", type=str, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=flux_transformer_default_batch_sizes,
    )
    args = parser.parse_args(args=args)
    export_flux_transformer_model_mlir(
        Path(args.parameters_path),
        output_path=args.output_path,
        batch_sizes=args.batch_sizes,
    )


if __name__ == "__main__":
    main()
