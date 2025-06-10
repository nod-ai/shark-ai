# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Merge multiple tuner-generated specs into a single one.

This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
It can be invoked in two ways:
    1. From another python script by importing and calling `merge_tuning_specs()`
    2. Directly from the command line to merge tuning spec files

Usage:
    python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
"""

import argparse
import logging

from iree.compiler import ir  # type: ignore

from .common import *

tune_logger = logging.getLogger("tune")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="merge_td_specs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs", nargs="+", help="Input MLIR tuning spec files to merge"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output path for merged MLIR file (if omitted, prints to stdout)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
    )

    args = parser.parse_args()
    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    tune_logger.addHandler(console_handler)

    with TunerContext() as tuner_ctx:
        td_specs = []
        for input_path in args.inputs:
            tune_logger.debug(f"Reading td spec: {input_path}")
            with open(input_path, "r") as f:
                td_spec_str = f.read()
                td_specs.append(ir.Module.parse(td_spec_str, tuner_ctx.mlir_ctx))

        # Emit warnings for duplicate matchers in td_specs.
        td_specs_to_link = determine_td_specs_to_link(td_specs, log_duplicates=True)
        merged_td_spec = link_tuning_specs(tuner_ctx, td_specs_to_link)
        if args.output:
            with open(args.output, "w") as f:
                f.write(str(merged_td_spec))
            tune_logger.debug(f"Merged spec written to: {args.output}")
        else:
            print(str(merged_td_spec))


if __name__ == "__main__":
    main()
