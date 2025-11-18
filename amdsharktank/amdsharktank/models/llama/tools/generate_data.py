# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates data files for calling iree-run-module from a prompt and config.

Usage:
  $ python -m amdsharktank.models.llama.tools.generate_data \
    --tokenizer=openlm-research/open_llama_3b_v2 \
    --config=/tmp/open-llama-3b-v2-f16.json \
    --output-dir=/tmp/inputs \
    --prompt="What is the meaning of life?"

  $ ls /tmp/inputs

    arg0.bin
    arg1.bin
    arg2.bin
    arg3.bin

  $ iree-run-module \
    --module=/tmp/open-llama-3b-v2-f16_cpu.vmfb \
    --parameters=model=/tmp/open-llama-3b-v2-f16.gguf \
    --function=prefill_bs4 \
    --device=local-task \
    --input=4x1xi64=@/tmp/inputs/arg0.bin \
    --input=4xi64=@/tmp/inputs/arg1.bin \
    --input=4x1xi64=@/tmp/inputs/arg2.bin \
    --input=1x2662400xf16=@/tmp/inputs/arg3.bin

# TODO(scotttodd): similar script to convert outputs to text via tokenizer
# TODO(scotttodd): teach service_v1_cli to also dump its inputs/outputs?
# TODO(scotttodd): generate expected outputs using reference model?
"""

from pathlib import Path
import logging
import sys
import json
import numpy as np

from transformers import LlamaTokenizer  # type: ignore

from ....utils.logging import get_logger
from .data_utils import write_ndarray_to_bin

logger = get_logger("amdsharktank.models.llama.tools.generate_data")


def main(argv):
    from ....utils import cli

    parser = cli.create_parser()
    parser.add_argument(
        "--tokenizer", help="name of hugginface tokenizer to use", required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="json config file with hyperparameters",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Generate .bin files into this directory",
        required=True,
    )
    parser.add_argument("--prompt", help="Prompt string", required=True)
    # TODO(scotttodd): output path (directory to dump .bin/.npy files)
    args = cli.parse(parser, args=argv)

    # Load config hyperparameters.
    with open(args.config) as f:
        config = json.load(f)
    logger.info("Loaded config with hyperparameters:")
    logger.info(json.dumps(config, indent=4))

    # Load tokenizer.
    # TODO(scotttodd): Unify tokenizer flags across amdsharktank and shortfin?
    #   cli.add_tokenizer_options(parser)
    #   tokenizer = cli.get_tokenizer(args)
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer, legacy=False)

    # TODO(scotttodd): loop over batch sizes (generate one dataset per batch size)
    prefill_batch_size = config["prefill_batch_sizes"][0]

    # Declare input arguments.
    # TODO(scotttodd): compute max_seq_len from tokens, _not_ config here
    arg0_prefill_tokens = np.zeros(
        [prefill_batch_size, config["max_seq_len"]], dtype=np.int64
    )
    arg1_prefill_seq_lens = np.zeros(prefill_batch_size, dtype=np.int64)
    # TODO(scotttodd): arg2 - attention block indices
    # TODO(scotttodd): arg3 - attention block buffer

    # Populate input arguments.
    # TODO(scotttodd): loop over 1 prompt per batch here (or duplicate)
    prompt = args.prompt
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
    logger.info(f"prompt -> encoded tokens: {prompt_tokens}")
    prompt_seq_len = len(prompt_tokens)
    arg0_prefill_tokens[0, 0:prompt_seq_len] = prompt_tokens
    arg1_prefill_seq_lens[0] = prompt_seq_len
    with np.printoptions(threshold=np.inf):
        logger.debug("arg0_prefill_tokens:")
        logger.debug(arg0_prefill_tokens)
        logger.debug("arg1_prefill_seq_lens:")
        logger.debug(arg1_prefill_seq_lens)

    logger.info(f"Writing argument .bin files to '{args.output_dir}'")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ndarray_to_bin(arg0_prefill_tokens, args.output_dir / "arg0.bin")
    write_ndarray_to_bin(arg1_prefill_seq_lens, args.output_dir / "arg1.bin")


if __name__ == "__main__":
    main(argv=sys.argv[1:])
