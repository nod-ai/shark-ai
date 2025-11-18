# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import torch

from amdsharktank.layers import *
from amdsharktank.types import *
from amdsharktank.models.llm import *
from amdsharktank.utils import cli
from amdsharktank.utils.attention import *


def main(args: list[str]):

    torch.no_grad().__enter__()

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    args = cli.parse(parser)

    dataset = cli.get_input_dataset(args)
    llama_config = LlamaModelConfig.from_dataset(dataset=dataset)
    llama_config.kv_cache_type = "paged"
    llama_config.activation_dtype = torch.float16
    model = PagedLlmModelV1(dataset.root_theta, llama_config)

    cache_state = model.cache.allocate(page_count=128)

    start_index = 0
    next_batch = torch.tensor(
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
            [
                1,
                1059,
                31871,
                1217,
                322,
                31871,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            + 48 * [0],
            [1, 573, 846, 1360, 1833, 2520, 2948, 5108, 4590] + 55 * [0],
            64 * [0],
        ]
    )
    assert next_batch.shape[1] % model.config.block_seq_stride == 0
    seq_block_ids = torch.tensor(
        [
            [127, 0, 0, 0],
            [126, 0, 0, 0],
            [125, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    # Important: Do not use a sequence length of 0 for empty batch slots
    # as it will cause softmax to nan due to a mask of all -inf. This then
    # propagates and causes badness.
    seq_lens = torch.tensor([12, 6, 9, 1])

    print(f"Step {start_index}")
    logits = model.prefill(
        next_batch,
        seq_lens=seq_lens,
        seq_block_ids=seq_block_ids,
        cache_state=cache_state,
    )
    # TODO: Normalize the output of extract_tokens_from_logits into tensor [bs, 1].
    tokens = torch.tensor(model.extract_tokens_from_logits(logits, seq_lens)).unsqueeze(
        1
    )
    print(f"  : tokens = {tokens}")
    print(f"  : cache[127] = {cache_state[0][127]}")
    print(f"  : cache[126] = {cache_state[0][126]}")
    print(f"  : cache[0] = {cache_state[0][0]}")
    print(f"  : cache[1] = {cache_state[0][1]}")

    # Decode a step.
    print("Decoding...")
    print(tokens.shape, tokens)
    start_positions = torch.tensor([12, 6, 0, 0])
    seq_lens = seq_lens + 1

    logits = model.decode(
        tokens,
        seq_lens=seq_lens,
        start_positions=start_positions,
        seq_block_ids=seq_block_ids,
        cache_state=cache_state,
    )
    tokens = torch.tensor(
        model.extract_tokens_from_logits(logits, [1, 1, 1, 1])
    ).unsqueeze(1)
    print(f"  : tokens = {tokens}")
    print(f"  : cache[127] = {cache_state[0][127]}")
    print(f"  : cache[126] = {cache_state[0][126]}")
    print(f"  : cache[0] = {cache_state[0][0]}")
    print(f"  : cache[1] = {cache_state[0][1]}")

    # from amdsharktank.models import llama
    # print(f"+++PREFILL XK = {llama.DEBUG_PREFILL_XK.shape}\n{llama.DEBUG_PREFILL_XK}")
    # print(f"+++DECODE  XK = {llama.DEBUG_DECODE_XK.shape}\n{llama.DEBUG_DECODE_XK}")
    # torch.testing.assert_close(llama.DEBUG_PREFILL_XK, llama.DEBUG_DECODE_XK)

    def save_prefill_module(model):
        from iree.compiler.extras.fx_importer import FxImporter
        from iree.compiler.ir import AsmState

        importer = FxImporter()
        # asm_state = AsmState(importer.module_op)

        print("Generating FX graph")

        class InferenceModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module("prefill", model)

            def forward(self, next_batch, attention_mask, seq_block_ids, *cache_state):
                return self.prefill.prefill(
                    next_batch,
                    attention_mask=attention_mask,
                    seq_block_ids=seq_block_ids,
                    cache_state=list(cache_state),
                )

        infmod = InferenceModule()
        prog = torch.export.export(
            infmod, (next_batch, attention_mask, seq_block_ids) + tuple(cache_state)
        )

        print(f"FX prog:", prog)
        importer.import_program(prog, func_name="prefill")
        output_file = "/tmp/prefill.mlirbc"
        print("Saving to:", output_file)
        with open(output_file, "wb") as f:
            importer.module_op.write_bytecode(f)

    # save_prefill_module()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
