# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import torch

from sharktank.layers import *
from sharktank.types import *
from sharktank.models.llm import *


def main(args: list[str]):
    from ..utils import cli

    torch.no_grad().__enter__()

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    args = cli.parse(parser)

    dataset = cli.get_input_dataset(args)
    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    llama_config = LlamaModelConfig(hp)
    llama_config.kv_cache_type = "direct"
    llama_config.activation_dtype = torch.float16
    model = PagedLlmModelV1(dataset.root_theta, llama_config)

    # bs ("batch size") == 1
    cache_state = model.cache.allocate(bs=1)

    start_index = 0
    tokens = torch.tensor(
        [
            [
                1,
                1059,
                31871,
                1217,
                322,
                266,
                3682,
                6075,
                31902,
                13,
                31849,
                31871,
                0,
                0,
                0,
                0,
            ]
            + 48 * [0],
        ]
    )
    assert tokens.shape[1] % model.cache.block_seq_stride == 0
    seq_block_ids = torch.tensor(
        [
            [127, 0, 0, 0],
        ]
    )

    # Important: Do not use a sequence length of 0 for empty batch slots
    # as it will cause softmax to nan due to a mask of all -inf. This then
    # propagates and causes badness.
    seq_lens = torch.tensor([12])

    attention_mask = model.attention_mask(
        model.input_mask(seq_lens, tokens.shape[1]),
    )

    print(f"Step {start_index}")
    logits = model.prefill(
        tokens,
        attention_mask=attention_mask,
        seq_block_ids=seq_block_ids,
        cache_state=cache_state,
    )
    # TODO: Normalize the output of extract_tokens_from_logits into tensor [bs, 1].
    tokens = torch.tensor(model.extract_tokens_from_logits(logits, seq_lens)).unsqueeze(
        1
    )
    print(f"  : tokens = {tokens}")

    # Decode a step.
    print("Decoding...")
    print(tokens.shape, tokens)
    start_positions = torch.tensor([12])
    seq_lens = seq_lens + 1
    decode_attention_mask = model.decode_attention_mask(
        model.input_mask(
            seq_lens,
            seq_block_ids.shape[1] * model.cache.block_seq_stride,
        ),
    )
    logits = model.decode(
        tokens,
        attention_mask=decode_attention_mask,
        start_positions=start_positions,
        seq_block_ids=seq_block_ids,
        cache_state=cache_state,
    )
    tokens = torch.tensor(model.extract_tokens_from_logits(logits, [1])).unsqueeze(1)
    print(f"  : tokens = {tokens}")

    def save_prefill_module(model):
        from iree.compiler.extras.fx_importer import FxImporter
        from iree.compiler.ir import AsmState

        importer = FxImporter()

        print("Generating FX graph")

        class InferenceModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module("prefill", model)

            def forward(self, tokens, attention_mask, seq_block_ids, *cache_state):
                return self.prefill.prefill(
                    tokens,
                    attention_mask=attention_mask,
                    seq_block_ids=seq_block_ids,
                    cache_state=list(cache_state),
                )

        infmod = InferenceModule()
        prog = torch.export.export(
            infmod, (tokens, attention_mask, seq_block_ids) + tuple(cache_state)
        )

        print(f"FX prog:", prog)
        importer.import_program(prog, func_name="prefill")
        output_file = "/tmp/prefill.mlirbc"
        print("Saving to:", output_file)
        with open(output_file, "wb") as f:
            importer.module_op.write_bytecode(f)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
