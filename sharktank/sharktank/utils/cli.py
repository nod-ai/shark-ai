# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Utilities for building command line tools."""

from typing import Optional, Sequence
import argparse
from pathlib import Path
import logging

import torch
from amdsharktank.types import Dataset, serialized_name_to_dtype
from . import hf_datasets, tokenizer


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
                dtype = serialized_name_to_dtype(dtype)
                assert isinstance(dtype, torch.dtype)
            setattr(parsed_args, attr, dtype)
    return parsed_args


def add_input_dataset_options(parser: argparse.ArgumentParser):
    """Adds options to load a GGUF dataset.

    Either the `--hf-dataset`, `--gguf-file`, `--irpa-file` or `--json-file` argument can be present.
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hf-dataset",
        help=f"HF dataset to use (available: {list(hf_datasets.ALL_DATASETS.keys())})",
    )
    group.add_argument("--gguf-file", type=Path, help="GGUF file to load")
    group.add_argument("--irpa-file", type=Path, help="IRPA file to load")
    group.add_argument("--json-file", type=Path, help="JSON file to load")


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
        choices=["decomposed", "torch", "amdsharktank"],
    )
    parser.add_argument(
        "--matmul-kernel",
        type=str,
        default="*",
        help="Matmul kernel selection. Can be a single kernel (e.g. 'amdsharktank.asm') or preference list with semicolon separator (e.g. 'amdsharktank.wave;amdsharktank.asm;*'). Use '*' to match any kernel.",
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
        "--pipeline-parallelism-size",
        type=int,
        default=1,
        help="Number of (roughly) uniform groups of layers to split the model for pipeline parallelism.",
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
    parser.add_argument(
        "--has-prefill-position",
        help="Determines if prefill offset supported during export",
        action="store_true",
    )
    parser.add_argument(
        "--use-extend-attention",
        help="Determines if extend attention is supported during export",
        action="store_true",
    )
    parser.add_argument(
        "--use-qk-norm",
        help="q and k got normalized in attention layer. for llama4",
        action="store_true",
    )
    parser.add_argument(
        "--use-toy-model",
        help="Generates toy model",
        action="store_true",
    )
    parser.add_argument(
        "--top-k",
        help="Export with a `top_k` kernel. If `top_k` == 1, argmax is exported."
        "Otherwise, `topk_k{k} is exported.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--top-k-chunk-size",
        help="Size of chunks to split into when exporting `top_k`.",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--use-linalgext-topk",
        action="store_true",
        help="Whether to use the linalg_ext topk implementation",
    )
    parser.add_argument(
        "--logits-normalization",
        default="none",
        help="Return the log softmax of the logits",
        choices=["none", "softmax", "log_softmax"],
    )
    parser.add_argument(
        "--prefill-final-logits",
        help="Return only the final logits",
        action="store_true",
    )
    parser.add_argument(
        "--attention-chunk-size",
        help="the size of each chunk used during chunked attention computation",
        type=int,
        default=None,
    )


def add_model_input_options(parser: argparse.ArgumentParser):
    """Adds input options for LLMs"""

    parser.add_argument(
        "--prompt",
        nargs="+",
        help="Custom prompt strings to run LLM or perplexity",
    )
    parser.add_argument(
        "--max-decode-steps",
        type=int,
        default=None,
        help="Maximum number of decode steps to perform.",
    )


def add_iree_flags(parser: argparse.ArgumentParser):
    """Adds IREE device flag options"""

    parser.add_argument(
        "--iree-device",
        type=str,
        action="append",
        help="List an IREE device from 'iree-run-module --list_devices'",
    )
    parser.add_argument(
        "--iree-hip-target",
        action="store",
        default="gfx942",
        help="Specify the iree-hip target version (e.g., gfx942)",
    )
    parser.add_argument(
        "--iree-hal-target-device",
        action="store",
        default="hip",
        help="Specify the iree-hal target device (e.g., hip, cpu)",
    )


def add_export_artifacts(parser: argparse.ArgumentParser):
    """Adds export & compile artifacts path options"""

    parser.add_argument(
        "--bs-prefill",
        help="Comma-separated batch size(s) to generate, e.g. `4` or `2,4`",
        type=lambda arg: [int(bs) for bs in arg.split(",")],
        default="4",
    )
    parser.add_argument(
        "--bs-decode",
        help="Comma-separated batch size(s) to generate, e.g. `4` or `2,4`",
        type=lambda arg: [int(bs) for bs in arg.split(",")],
        default="4",
    )
    parser.add_argument(
        "--strict",
        help="Enables strictness during export",
        action="store_true",
    )
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        type=str,
    )
    parser.add_argument(
        "--output-config",
        help="Output file path for exported config file",
        type=str,
    )
    parser.add_argument(
        "--output-vmfb",
        help="Output file path for compiled vmfb file",
        type=str,
    )
    parser.add_argument(
        "--extra-compile-arg",
        help=(
            "Additional flag(s) to provide to the IREE compiler. "
            "E.g. `--extra-compile-arg=--compile-mode=vm --extra-compile-arg=--iree-vm-target-extension-f32`"
        ),
        action="append",
        type=str,
        default=[],
    )


def add_save_tensor_options(parser: argparse.ArgumentParser):
    """Adds options to save input and intermediate tensors to separate files"""

    parser.add_argument(
        "--save_intermediates_path",
        help="save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors",
    )
    parser.add_argument(
        "--dump-path",
        help="Path to dump prefill/decode input tensors to npy files",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dump-decode-steps",
        help="Number of decode steps to dump decode input tensors",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prompt-seq-len",
        help="Seq len to generate input prompts for prefill",
        type=int,
    )
    parser.add_argument(
        "--bs",
        help="Batch size",
        type=int,
        default="4",
    )


def add_quantization_options(parser: argparse.ArgumentParser):
    """Adds quantization options"""

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


def add_log_options(parser: argparse.ArgumentParser):
    """Adds log options"""

    parser.add_argument(
        "--verbose",
        help="Include verbose logging",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )


def add_evaluate_options(parser: argparse.ArgumentParser):
    """Adds input text options for evaluate/perplexity"""

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=128,
        help="Number of prompts/batch size for perplexity test (1 to 128)",
    )
    parser.add_argument(
        "--prompt-list",
        nargs="+",
        type=str,
        help="Custom prompts to run perplexity",
    )
    parser.add_argument(
        "--prefill-length",
        type=int,
        default=None,
        help="Number of tokens for prefill before starting decode.",
    )


def get_dtype_flags(args) -> dict[str, str]:
    return {
        "activation_dtype": args.activation_dtype,
        "attention_dtype": args.attention_dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
    }


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
    elif args.json_file is not None:
        return {"json": [args.json_file]}


def get_input_dataset(args) -> Dataset:
    """Loads and returns a dataset from the given args.

    Presumes that the arg parser was initialized with |add_input_dataset|.
    """
    data_files = get_input_data_files(args)
    device = getattr(args, "device", None)

    if "gguf" in data_files:
        return Dataset.load(data_files["gguf"][0], file_type="gguf", device=device)

    if "irpa" in data_files:
        return Dataset.load(data_files["irpa"][0], file_type="irpa", device=device)

    if "json" in data_files:
        return Dataset.load(data_files["json"][0], file_type="json", device=device)

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
