# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from amdsharktank.utils import cli
from amdsharktank.utils.hf import import_hf_dataset_from_hub


def main(argv: list[str] | None = None):

    parser = cli.create_parser(
        description=(
            "Import a Hugging Face dataset. "
            "This includes downloading from the HF hub and transforming the model "
            "parameters into an IRPA. "
            "The import can either be specified through the various detailed "
            "parameters or a preset name can be reference. "
            "The HF dataset details can also be reference by a named preregistered dataset. "
            "The HF dataset is only concerned with what files need to be downloaded from the hub."
        )
    )
    parser.add_argument(
        "--output-irpa-file",
        type=Path,
        help="IRPA file to save dataset to",
    )
    parser.add_argument(
        "repo_id_or_path",
        nargs="?",
        default=None,
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
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help=(
            "A name of a preset HF dataset. "
            "This is mutually exclusive with specifying repo id or a model path."
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "A name of a preset to import. "
            "This is different form a preset HF dataset, which only specifies the HF "
            "files to download. "
            "The import preset also specifies the import transformations. "
            "When using a preset the output directory structure would be relative to "
            "the current working directory."
        ),
    )
    args = cli.parse(parser, args=argv)

    import_hf_dataset_from_hub(
        args.repo_id_or_path,
        revision=args.revision,
        subfolder=args.subfolder,
        config_subpath=args.config_subpath,
        output_irpa_file=args.output_irpa_file,
        hf_dataset=args.hf_dataset,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
