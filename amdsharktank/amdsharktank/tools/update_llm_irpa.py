# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import logging

from amdsharktank.layers.configs.llm_configs import LlamaModelConfig
from amdsharktank.utils import cli

logger = logging.getLogger(__name__)


def main():

    # Set up logging

    parser = cli.create_parser()
    parser.add_argument(
        "--output", type=Path, help="Save the output file", required=True
    )

    parser.add_argument(
        "--interleave-rotary",
        help="Set to interleave rotary embedding",
        description="This is typically when the model comes from GGUF which uses interleaved embedding representations",
        action="store_true",
    )
    parser.add_argument(
        "--concatenate-rotary",
        help="Set to interleave rotary embedding",
        description="This is typically when the model comes from huggingface which uses concatenated embedding representations",
        action="store_true",
    )

    cli.add_input_dataset_options(parser)
    cli.add_log_options(parser)

    args = cli.parse(parser)
    config = cli.get_input_dataset(args)

    props = LlamaModelConfig.from_properties(config.properties)

    if args.interleave_rotary and args.concatenate_rotary:
        raise ValueError("Cannot set both --interleave-rotary and --concatenate-rotary")

    if args.interleave_rotary:
        props.hp.rope_interleave_emb = True

    if args.concatenate_rotary:
        props.hp.rope_interleave_emb = False

    config.properties = props.hp.to_gguf_props()
    logger.setLevel(args.loglevel)
    config.save(args.output, file_type="irpa")


if __name__ == "__main__":
    main()
