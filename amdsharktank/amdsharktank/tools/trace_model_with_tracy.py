# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import sys
import json
import os
import subprocess
from copy import copy

from ..layers import ModelConfig
from ..utils.iree import run_model_with_iree_run_module


def main(args: list[str] | None = None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Trace an already exported and compiled model from a model config."
    )
    parser.add_argument(
        "--config", type=str, default="-", help="Path to the model config."
    )
    parser.add_argument(
        "--function", type=str, required=True, help="The function to trace."
    )
    args = parser.parse_args(args)

    if args.config == "-":
        config = json.load(sys.stdin)
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

    env = copy(os.environ)
    env["IREE_PY_RUNTIME"] = "tracy"
    run_model_with_iree_run_module(
        ModelConfig.create(**config), function=args.function, env=env
    )


if __name__ == "__main__":
    main()
