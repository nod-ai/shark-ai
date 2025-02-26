# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Utilities for building command line tools."""

from typing import Dict, Optional, Sequence

import argparse
import logging
from pathlib import Path
import torch
from ..types import Dataset

from . import hf_datasets
from . import tokenizer


def create_parser(
    *,
    prog: Optional[str] = None,
    usage: Optional[str] = None,
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, usage=usage, description=description)
    return parser


def parse(parser: argparse.ArgumentParser, *, args: Sequence[str] | None = None):
    """Parses arguments and does any prescribed global process setup."""
    parsed_args = parser.parse_args(args)
    # Set torch dtypes
    for attr in ["activation_dtype", "attention_dtype", "kv_cache_dtype"]:
        if hasattr(parsed_args, attr):
            dtype = getattr(parsed_args, attr)
            if dtype is not None:
                dtype = getattr(torch, dtype)
                assert isinstance(dtype, torch.dtype)
            setattr(parsed_args, attr, dtype)
    return parsed_args


def add_input_dataset_options(parser: argparse.ArgumentParser):
    """Adds options to load a GGUF dataset.

    Either the `--hf-dataset`, `--gguf-file`, or `--irpa-file` argument can be present.
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hf-dataset",
        help=f"HF dataset to use (available: {list(hf_datasets.ALL_DATASETS.keys())})",
    )
    group.add_argument("--gguf-file", type=Path, help="GGUF file to load")
    group.add_argument("--irpa-file", type=Path, help="IRPA file to load")


def add_output_dataset_options(parser: argparse.ArgumentParser):
    """Adds options to save a dataset.

    This will result in the --output-irpa-file argument being added.
    """
    parser.add_argument(
        "--output-irpa-file",
        type=Path,
        required=True,
        help="IRPA file to save dataset to",
    )


def add_model_options(parser: argparse.ArgumentParser):
    """Adds model config options not exclusive to export or eager"""
    parser.add_argument(
        "--attention-kernel",
        type=str,
        default="torch",
        choices=["decomposed", "torch"],
    )
    parser.add_argument(
        "--skip-prefill",
        help="Skips exporting prefill",
        action="store_true",
    )
    parser.add_argument(
        "--skip-decode",
        help="Skips exporting decode",
        action="store_true",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--activation-dtype",
        help="DType to use for activations in the model",
        default="float16",
    )
    parser.add_argument(
        "--attention-dtype",
        help="DType to use for attention in the model",
        default="float16",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        help="DType to use for the KV cache. If not given will be attention dtype",
        default=None,
    )
    parser.add_argument("--device", help="Torch device (or default)")

    parser.add_argument(
        "--tensor-parallelism-size",
        type=int,
        default=1,
        help="Number of devices for tensor parallel sharding. Will be overridden by dataset.properties if present",
    )
    parser.add_argument(
        "--block-seq-stride",
        help="Block sequence stride for paged KV cache, must divide evenly into the context length",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device-block-count",
        help="Block per device for paged KV cache",
        type=int,
        default=512,
    )


def add_quantization_options(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--fake-quant",
        action=argparse.BooleanOptionalAction,
        help="whether or not to run/export the model in fake quant mode. Note, running eagerly without fake quant is dependent on torch types supporting operations. YMMV",
    )


def add_tokenizer_options(parser: argparse.ArgumentParser):
    """Adds options for specifying a tokenizer.

    All are optional and if not specified, some default options will be taken
    based on the dataset.
    """
    parser.add_argument(
        "--tokenizer-type", help="Tokenizer type or infer from dataset if not specified"
    )
    parser.add_argument(
        "--tokenizer-config-json",
        help="Direct path to a tokenizer_config.json file",
        type=Path,
    )


def get_input_data_files(args) -> Optional[dict[str, list[Path]]]:
    """Gets data files given the input arguments.

    Keys may contain:
      * tokenizer_config.json
      * gguf
      * irpa
    """
    if args.hf_dataset is not None:
        dataset = hf_datasets.get_dataset(args.hf_dataset).download()
        return dataset
    elif args.gguf_file is not None:
        return {"gguf": [args.gguf_file]}
    elif args.irpa_file is not None:
        return {"irpa": [args.irpa_file]}


def get_input_dataset(args) -> Dataset:
    """Loads and returns a dataset from the given args.

    Presumes that the arg parser was initialized with |add_input_dataset|.
    """
    data_files = get_input_data_files(args)
    if "gguf" in data_files:
        return Dataset.load(data_files["gguf"][0], file_type="gguf")

    if "irpa" in data_files:
        return Dataset.load(data_files["irpa"][0], file_type="irpa")

    raise ValueError(f'Dataset format unsupported. Must be "gguf" or "irpa".')


def get_tokenizer(args) -> tokenizer.InferenceTokenizer:
    """Gets a tokenizer based on arguments.

    If the data_files= dict is present and explicit tokenizer options are not
    set, we will try to infer a tokenizer from the data files.
    """
    if args.tokenizer_type == "fake":
        return tokenizer.fake_tokenizer()

    if args.tokenizer_config_json is not None:
        data_files = {"tokenizer_config.json": [args.tokenizer_config_json]}
    else:
        data_files = get_input_data_files(args)

    tokenizer_type = args.tokenizer_type
    if tokenizer_type is None:
        if "tokenizer_config.json" in data_files:
            return tokenizer.load_tokenizer(
                data_files["tokenizer_config.json"][0].parent,
                tokenizer_type="transformers",
            )
        else:
            raise ValueError(f"Could not infer tokenizer from data files: {data_files}")
    else:
        raise ValueError(f"Unsupported --tokenizer-type argument: {tokenizer_type}")
