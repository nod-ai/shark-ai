# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import json
import torch

from typing import Optional

import torch.nn.functional as F

from iree.turbine.aot import *

from sharktank.layers import *
from sharktank.types import *
from sharktank.models.llama.testing import *
from sharktank.utils import cli
from sharktank.utils.create_cache import *

# TODO: Should be using a base class with the protocol supported.


def paged_attention(
    attention_block: PagedLlamaAttentionBlock,
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    is_causal: bool,
    seq_block_ids: torch.Tensor,
    start_positions: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cache_state: list[torch.Tensor] = None,
):

    block_index = attention_block.block_index
    head_count = attention_block.head_count
    bs, batch_seq_len, _, _ = xq.shape

    # Full sequence length.
    kv_seq_len = seq_block_ids.shape[1] * attention_block.cache.block_seq_stride

    if start_positions is None:
        attn_output = paged_attention.forward_prefill(
            q=xq,
            k=xk,
            v=xv,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            block_index=block_index,
            head_count_attn=head_count,
            mask=attention_mask,
        )
    else:
        attn_output = paged_attention.forward_decode(
            q=xq,
            k=xk,
            v=xv,
            cache_state=cache_state,
            seq_block_ids=seq_block_ids,
            block_index=block_index,
            kv_seq_len=kv_seq_len,
            start_positions=start_positions,
            head_count_attn=head_count,
            mask=attention_mask,
        )

    attn_output = attn_output.transpose(1, 2).reshape(bs, batch_seq_len, -1)
    return attn_output


def run_llama(
    model: PagedLlamaAttentionBlock,
    config: LlamaModelConfig,
    phase: str,
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    # [1, 1, batch_seq_len, batch_seq_len]
    attention_mask: torch.Tensor,
    # [bs, batch_seq_len // block_seq_stride]
    seq_block_ids: torch.Tensor,
    cache_state: list[torch.Tensor],
    # [bs] of starting positions
    start_positions: Optional[torch.Tensor] = None,
):

    if phase not in ["prefill", "decode"]:
        raise ValueError("'phase' argument needs to be either 'prefill' or 'decode'")

    h = paged_attention(
        model,
        xq=xq,
        xk=xk,
        xv=xv,
        is_causal=config.is_causal,
        start_positions=start_positions,
        attention_mask=attention_mask,
        cache_state=cache_state,
        seq_block_ids=seq_block_ids,
    )

    return h


def main():

    parser = cli.create_parser()
    # cli.add_input_dataset_options(parser)
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        default="/tmp/sharktank/artifacts/paged_llama.mlir",
    )
    parser.add_argument(
        "--output-config",
        help="Output file path for exported config file",
        default="/tmp/sharktank/artifacts/paged_llama.json",
    )
    parser.add_argument(
        "--bs",
        help="Comma-separated batch size(s) to generate, e.g. `4` or `2,4`",
        type=lambda arg: [int(bs) for bs in arg.split(",")],
        default="4",
    )
    parser.add_argument(
        "--verbose",
        help="Include verbose logging",
        action="store_true",
    )

    parser.add_argument(
        "--is-causal",
        help="Enable Causal attention",
        action="store_true",
    )
    # TODO: move this to CLI to enable re-use with eager
    parser.add_argument(
        "--attention_kernel",
        help="decomposed/torch",
        default="decomposed",
    )

    args = cli.parse(parser)

    hp = configs.LlamaHParams(
        context_length=4096,
        embedding_length=4096,
        block_count=1,
        feed_forward_length=11008,
        attn_head_dim=128,
        rope_dimension_count=128,
        attention_head_count=32,
        attention_layer_norm_rms_epsilon=9.999999747378752e-06,
        attention_head_count_kv=32,
        model_arch="llama",
    )

    llama_config = LlamaModelConfig(hp)
    llama_config.kv_cache_type = "paged"
    llama_config.bs = args.bs
    llama_config.is_causal = args.is_causal

    attention_block_theta = make_attention_block_theta(
        feature_dim=llama_config.hp.attention_head_count
        * llama_config.hp.attn_head_dim,
        ffn_dim=llama_config.hp.feed_forward_length,
        dtype=llama_config.attention_dtype,
    )

    causal_model = causal_llm.BaseCausalLMModel(
        attention_block_theta, context_length=llama_config.hp.context_length
    )

    model = PagedLlamaAttentionBlock(
        theta=attention_block_theta,
        block_index=0,
        cache=create_paged_kv_cache(llama_config),
        head_count=llama_config.hp.attention_head_count,
        head_dim=llama_config.hp.attn_head_dim,
        head_count_kv=llama_config.hp.attention_head_count_kv,
        rms_epsilon=llama_config.hp.attention_layer_norm_rms_epsilon,
        attention_kernel=args.attention_kernel,
    )

    def generate_params_json(hp, prefill_bs: list[int], decode_bs: list[int]):
        return {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": hp.context_length,
            "attn_head_count": hp.attention_head_count,
            "attn_head_dim": hp.attn_head_dim,
            "prefill_batch_sizes": prefill_bs,
            "decode_batch_sizes": decode_bs,
            "transformer_block_count": hp.block_count,
            "block_seq_stride": llama_config.block_seq_stride,
        }

    fxb = FxProgramsBuilder(model)

    def generate_batch_prefill(bs: int):
        tokens = torch.empty(bs, 64, dtype=torch.int64)
        seq_lens = torch.empty(bs, dtype=torch.int64)
        seq_block_ids = torch.empty(bs, 4, dtype=torch.int64)
        block_dim = torch.export.Dim(
            "block", max=(hp.context_length - 1) // llama_config.block_seq_stride
        )
        sl_dim = llama_config.block_seq_stride * block_dim

        if llama_config.kv_cache_type == "paged":
            cache_state = model.cache.allocate(
                page_count=hp.context_length // llama_config.block_seq_stride
            )
            page_dim = torch.export.Dim("page")
            cache_state_dynamic_shapes = [{0: page_dim}]
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

        dynamic_shapes = {
            "tokens": {1: sl_dim},
            "seq_lens": {},
            "seq_block_ids": {1: block_dim},
            "cache_state": cache_state_dynamic_shapes,
        }

        q = torch.zeros((bs, 64, 32, 128), dtype=torch.float16)
        k = torch.zeros((bs, 64, 32, 128), dtype=torch.float16)
        v = torch.zeros((bs, 64, 32, 128), dtype=torch.float16)

        print(f"Exporting prefill_bs{bs}")
        example_args = (q, k, v, seq_lens, seq_block_ids, cache_state)

        @fxb.export_program(
            name=f"prefill_bs{bs}",
            args=example_args,
        )
        def _(model, q, k, v, seq_lens, seq_block_ids, cache_state):

            if llama_config.is_causal:
                attention_mask = None
            else:
                sl = tokens.shape[1]
                input_mask = causal_model.input_mask(seq_lens, sl)
                attention_mask = causal_model.attention_mask(input_mask)

            h = run_llama(
                model=model,
                config=llama_config,
                phase="prefill",
                xq=q,
                xk=k,
                xv=v,
                attention_mask=attention_mask,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )
            return h

    def generate_batch_decode(bs: int):
        tokens = torch.ones(bs, 1, dtype=torch.int64)
        seq_lens = torch.ones(bs, dtype=torch.int64)
        start_positions = torch.ones(bs, dtype=torch.int64)
        seq_block_ids = torch.zeros(bs, 4, dtype=torch.int64)
        block_dim = torch.export.Dim(
            "block", max=(hp.context_length - 1) // llama_config.block_seq_stride
        )

        if llama_config.kv_cache_type == "paged":
            cache_state = model.cache.allocate(
                page_count=hp.context_length // llama_config.block_seq_stride
            )
            page_dim = torch.export.Dim("page")
            cache_state_dynamic_shapes = [{0: page_dim}]
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

        dynamic_shapes = {
            "tokens": {},
            "seq_lens": {},
            "start_positions": {},
            "seq_block_ids": {1: block_dim},
            "cache_state": cache_state_dynamic_shapes,
        }

        q = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)
        k = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)
        v = torch.zeros((bs, 1, 32, 128), dtype=torch.float16)

        print(f"Exporting decode_bs{bs}")
        example_args = (q, k, v, seq_lens, start_positions, seq_block_ids, cache_state)

        @fxb.export_program(
            name=f"decode_bs{bs}",
            args=example_args,
        )
        def _(
            model,
            q,
            k,
            v,
            seq_lens,
            start_positions,
            seq_block_ids,
            cache_state,
        ):

            if llama_config.is_causal:
                attention_mask = None
            else:
                input_mask = causal_model.input_mask(
                    seq_lens, seq_block_ids.shape[1] * model.cache.block_seq_stride
                )
                attention_mask = causal_model.decode_attention_mask(input_mask)

            h = run_llama(
                model=model,
                config=llama_config,
                phase="decode",
                xq=q,
                xk=k,
                xv=v,
                attention_mask=attention_mask,
                start_positions=start_positions,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )

            return h

    bsizes = []
    for bs in llama_config.bs:
        generate_batch_prefill(bs)
        generate_batch_decode(bs)
        bsizes.append(bs)

    if args.verbose:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    config = generate_params_json(hp, bsizes, bsizes)
    print("GENERATED!")

    print("Exporting")
    output = export(fxb)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)
    json.dump(config, open(args.output_config, "w"))


if __name__ == "__main__":
    main()
