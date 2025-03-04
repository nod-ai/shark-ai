# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from argparse import ArgumentParser
from typing import Optional
from pathlib import Path

from .testing import export_clip_toy_text_model_default_iree_test_data


def main(args: Optional[list[str]] = None):
    parser = ArgumentParser(
        description=(
            "Export test data for toy-sized CLIP text model."
            " This includes the program MLIR, parameters, sample input and expected output."
            " Exports float32 and bfloat16 model variants."
            " The expected output is always in float32 precision."
        )
    )
    parser.add_argument("--output-dir", type=str, default=f"clip_toy_text_model")
    args = parser.parse_args(args=args)
    export_clip_toy_text_model_default_iree_test_data(Path(args.output_dir))


if __name__ == "__main__":
    main()
