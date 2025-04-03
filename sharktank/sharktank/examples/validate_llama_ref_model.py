# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import torch

from sharktank.layers import *
from sharktank.types import *
from sharktank.models.llama.llama_ref import *
from sharktank.utils import cli


def main(args: list[str]):

    torch.no_grad().__enter__()

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    args = cli.parse(parser)

    dataset = cli.get_input_dataset(args)
    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    ref_llama_config = RefLlamaModelConfig(hp)
    ref_llama_config.activation_dtype = torch.float16
    model = DirectCacheLlamaModelV1(dataset.root_theta, ref_llama_config)

    kv_cache = model.create_cache(bs=1)
    start_index = 0
    next_tokens = [1, 1059, 31871, 1217, 322, 266, 3682, 6075, 31902, 13, 31849, 31871]
    print(f"Step {start_index}")
    tokens = model.forward(
        torch.tensor([next_tokens]), start_index=start_index, local_kv_cache=kv_cache
    )
    print(f"  : tokens = {tokens}")

    # Decode a step.
    print("Decoding...")
    print(tokens.shape, tokens)
    decode_token = model.forward(tokens, start_index=12, local_kv_cache=kv_cache)
    print(f"  : decode tokens = {decode_token}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
