# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

from amdsharktank.layers import *
from amdsharktank.types import *
from amdsharktank.utils import cli
from amdsharktank.utils.logging import get_logger

logger = get_logger("amdsharktank.tools.dump_theta_json")


def main():

    # Set up logging

    parser = cli.create_parser()
    parser.add_argument(
        "--output-json", type=Path, help="Save the theta json to this file"
    )

    cli.add_input_dataset_options(parser)
    cli.add_log_options(parser)

    args = cli.parse(parser)
    config = cli.get_input_dataset(args)

    logger.setLevel(args.loglevel)

    config.save(args.output_json, file_type="json")


if __name__ == "__main__":
    main()
