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
import subprocess
import tempfile
import os

from iree.compiler import ir  # type: ignore

from .common import *

tune_logger = logging.getLogger("tune")


def combine_tuning_specs(
    tuner_ctx: TunerContext, td_specs: list[ir.Module]
) -> ir.Module:
    """
    Puts multiple input modules `td_specs` into a single top-level container module.
    This function does *not* attempt to merge or link `td_specs` across modules.
    """
    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
        top_module = ir.Module.create()
        top_module.operation.attributes[
            "transform.with_named_sequence"
        ] = ir.UnitAttr.get()

        for td_spec in td_specs:
            top_module.body.append(td_spec.operation.clone())
        return top_module


def merge_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
    """
    Merges multiple input modules (`td_specs`) into a single tuning specification module.
    First, the input modules are combined into a container module. Then, the external
    `iree-opt` tool is invoked with the `--iree-codegen-link-tuning-specs` pass to
    link or merge the individual tuning specs. When all input specs are marked with the
    default attribute `iree_codegen.tuning_spec_with_default_entrypoint`, they are merged
    into one tuning spec.
    """
    module = combine_tuning_specs(tuner_ctx, td_specs)
    iree_opt = ireec.binaries.find_tool("iree-opt")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "tmp_input.mlir")
        output_path = os.path.join(tmpdir, "tmp_output.mlir")

        with open(input_path, "w") as f:
            f.write(str(module))

        result = subprocess.run(
            [
                iree_opt,
                "--iree-codegen-link-tuning-specs",
                input_path,
                "-o",
                output_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"iree-opt failed: {result.stderr}")

        with open(output_path, "r") as f:
            output_mlir = f.read()
            return ir.Module.parse(output_mlir, tuner_ctx.mlir_ctx)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="merge_td_specs",
        description="""
            Merge multiple tuner-generated specs into a single one.

            This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
            It can be invoked in two ways:
                1. From another python script by importing and calling `merge_tuning_specs()`
                2. Directly from the command line to merge tuning spec files

            Usage:
                python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
        """,
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

        merged_td_spec = merge_tuning_specs(tuner_ctx, td_specs)
        if args.output:
            with open(args.output, "w") as f:
                f.write(str(merged_td_spec))
            tune_logger.debug(f"Merged spec written to: {args.output}")
        else:
            print(str(merged_td_spec))


if __name__ == "__main__":
    main()
