# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from sharktank.utils import cli
from sharktank.utils.hf import import_hf_dataset_from_hub


def main(argv: list[str] | None = None):

    parser = cli.create_parser(description="Import a Hugging Face dataset.")
    cli.add_output_dataset_options(parser)
    parser.add_argument(
        "repo_id_or_path",
        type=str,
        help='Local path to the model or Hugging Face repo id, e.g. "meta-llama/Meta-Llama-3-8B-Instruct"',
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Git SHA, tag or branch.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Subfolder for the repo. For example black-forest-labs/FLUX.1-schnell has a text encoder in a subfolder text_encoder",
    )
    parser.add_argument(
        "--config-subpath",
        type=str,
        default=None,
        help="Subpath inside the subfolder for the model config file. Defaults to config.json.",
    )
    args = cli.parse(parser, args=argv)

    import_hf_dataset_from_hub(
        args.repo_id_or_path,
        revision=args.revision,
        subfolder=args.subfolder,
        config_subpath=args.config_subpath,
        output_irpa_file=args.output_irpa_file,
    )


if __name__ == "__main__":
    main()
