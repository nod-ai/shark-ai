# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import json
from typing import Any, Dict
import torch

from iree.turbine.aot import *

from sharktank.layers import *
from sharktank.types import *
from sharktank.utils.math import ceildiv

# TODO: Should be using a base class with the protocol supported.
from ..models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
from ..models.llama.sharding import shard_theta
from ..models.mixtral.mixtral import *
from ..models.grok.grok import *
from .. import ops


def main():
    from ..utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        default="/tmp/batch_llama_v1.mlir",
    )
    parser.add_argument(
        "--output-config",
        help="Output file path for exported config file",
        default="/tmp/batch_llama_v1.json",
    )
    parser.add_argument(
        "--bs",
        help="Comma-separated batch size(s) to generate, e.g. `4` or `2,4`",
        type=lambda arg: [int(bs) for bs in arg.split(",")],
        default="4",
    )
    parser.add_argument(
        "--block-seq-stride",
        help="Block sequence stride for paged KV cache, must divide evenly into the context length",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--verbose",
        help="Include verbose logging",
        action="store_true",
    )
    parser.add_argument(
        "--strict",
        help="Enables strictness during export",
        action="store_true",
    )

    cli.add_quantization_options(parser)
    cli.add_model_options(parser)
    args = cli.parse(parser)
    dataset_type = cli.get_input_data_files(args)
    dataset_type = "irpa" if "irpa" in dataset_type else "gguf"
    dataset = cli.get_input_dataset(args)
    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    tensor_parallelism_size = (
        dataset.properties["tensor_parallelism_size"]
        if "tensor_parallelism_size" in dataset.properties
        else 1
    )

    llama_config = LlamaModelConfig(
        hp,
        tensor_parallelism_size=tensor_parallelism_size,
        use_hf=False,
        static_tables=False,  # Rely on the compiler for hoisting tables.
        kv_cache_type="paged",
        attention_kernel=args.attention_kernel,
        block_seq_stride=args.block_seq_stride,
    )
    llama_config.fake_quant = args.fake_quant

    if llama_config.hp.expert_count:
        if llama_config.hp.model_arch == "grok":
            model = PagedGrokModelV1(dataset.root_theta, llama_config)
        else:
            model = PagedMixtralModelV1(dataset.root_theta, llama_config)
    else:
        model = PagedLlamaModelV1(dataset.root_theta, llama_config)

    def generate_params_json(
        hp: LlamaHParams, prefill_bs: list[int], decode_bs: list[int]
    ) -> Dict[str, Any]:
        """
        Generate config.json for shortfin.


        For shortfin, we only write attention_head_count_kv because that's all shortfin needs.
        Note that this is different from hp.attn_head_count when grouped attention shares kvcache between heads.
        """
        return {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": hp.context_length,
            "attn_head_dim": hp.attn_head_dim,
            "prefill_batch_sizes": prefill_bs,
            "decode_batch_sizes": decode_bs,
            "transformer_block_count": hp.block_count,
            "paged_kv_cache": {
                "attention_head_count_kv": hp.attention_head_count_kv,
                "block_seq_stride": llama_config.block_seq_stride,
                "device_block_count": 256,  # so that this makes its way into the config file & can be edited.
            },
        }

    # Unrolling cache updates by batch row makes dynamo sad without an
    # override. There may be a better way to do this.
    import torch._dynamo.config as dynamo_config

    # TODO: Seems removed from 2.3+
    # dynamo_config.max_loop_unroll_nodes = 0

    fxb = FxProgramsBuilder(model)

    def setup_cache(model, shard_count):
        if model.config.kv_cache_type == "paged":
            cache_state = model.cache.allocate(
                page_count=hp.context_length // llama_config.block_seq_stride
            )
            page_dim = torch.export.Dim("page")

            dynamic_shapes = [{0: page_dim}]
            unpacked = cache_state
            arg_affinities = {}
            shard_dim = None

            # Need to unpacke that state when sharded
            if llama_config.tensor_parallelism_size > 1:
                shard_dim = cache_state[0].shard_dim

                unpacked = [[shard._data for shard in cs.shards] for cs in cache_state]
                dynamic_shapes = [
                    [ds] * llama_config.tensor_parallelism_size for ds in dynamic_shapes
                ]

                for i in range(llama_config.tensor_parallelism_size):
                    arg_affinities[i] = DeviceAffinity(str(i))

            return unpacked, shard_dim, dynamic_shapes, arg_affinities

        elif model.config.kv_cache_type == "direct":
            cache_state = model.cache.allocate(bs=1)
            # Direct cache dimensions:
            #   2 * transformer_block_count of...
            #   [bs, seq_length, attn_head_count, attn_head_dim]
            dynamic_shapes = [None]
            arg_affinities = {}
            shard_dim = None
            return torch.stack(cache_state), shard_dim, dynamic_shapes, arg_affinities
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

    def repack_cache(cache, shard_dim):
        return [SplitPrimitiveTensor(ts=c, shard_dim=shard_dim) for c in cache]

    def generate_batch_prefill(bs: int):
        # torch.export.Dim would make min at least 2
        block_dim_min = 2
        block_dim_max = ceildiv(hp.context_length, llama_config.block_seq_stride) - 1
        block_dim = torch.export.Dim("block", min=block_dim_min, max=block_dim_max)
        sl_dim = llama_config.block_seq_stride * block_dim
        seq_block_ids = torch.empty(bs, block_dim_min, dtype=torch.int64)
        tokens = torch.empty(
            bs,
            seq_block_ids.shape[1] * llama_config.block_seq_stride,
            dtype=torch.int64,
        )
        seq_lens = torch.empty(bs, dtype=torch.int64)

        cache, cache_shard_dim, cache_dynamic_shapes, arg_affinities = setup_cache(
            model, llama_config.tensor_parallelism_size
        )

        if llama_config.tensor_parallelism_size > 1:
            # We need to offset the indices for the cache
            arg_affinities = {key + 3: arg_affinities[key] for key in arg_affinities}

            for i in range(3):
                arg_affinities[i] = DeviceAffinity("0")

        dynamic_shapes = {
            "tokens": {1: sl_dim},
            "seq_lens": {},
            "seq_block_ids": {1: block_dim},
            "cs": cache_dynamic_shapes,
        }

        print(f"Exporting prefill_bs{bs}")

        @fxb.export_program(
            name=f"prefill_bs{bs}",
            args=(tokens, seq_lens, seq_block_ids, cache),
            dynamic_shapes=dynamic_shapes,
            strict=args.strict,
            arg_device=arg_affinities,
        )
        def _(model, tokens, seq_lens, seq_block_ids, cs):
            if (
                model.config.tensor_parallelism_size == 1
                and model.config.kv_cache_type == "direct"
            ):
                cache_tensors = torch.unbind(cs)
            else:
                cache_tensors = cs

            sl = tokens.shape[1]
            input_mask = model.input_mask(seq_lens, sl)
            attention_mask = model.attention_mask(input_mask)

            if llama_config.tensor_parallelism_size != 1:
                shard_count = llama_config.tensor_parallelism_size

                tokens = ops.replicate(tokens, count=shard_count)
                attention_mask = ops.replicate(attention_mask, count=shard_count)
                seq_block_ids = ops.replicate(seq_block_ids, count=shard_count)

                cache_tensors = repack_cache(cs, cache_shard_dim)

            logits = model.prefill(
                tokens,
                attention_mask=attention_mask,
                seq_block_ids=seq_block_ids,
                cache_state=cache_tensors,
            )

            if llama_config.tensor_parallelism_size != 1:
                logits = ops.unshard(logits)

            return logits

    def generate_batch_decode(bs: int):
        # torch.export.Dim would make min at least 2
        block_dim_min = 2
        block_dim_max = ceildiv(hp.context_length, llama_config.block_seq_stride) - 1
        block_dim = torch.export.Dim("block", min=block_dim_min, max=block_dim_max)
        tokens = torch.empty(
            bs,
            1,
            dtype=torch.int64,
        )
        seq_lens = torch.empty(bs, dtype=torch.int64)
        start_positions = torch.ones(bs, dtype=torch.int64)
        seq_block_ids = torch.empty(bs, block_dim_min, dtype=torch.int64)

        (
            cache_state,
            cache_shard_dim,
            cache_dynamic_shapes,
            arg_affinities,
        ) = setup_cache(model, llama_config.tensor_parallelism_size)

        if llama_config.tensor_parallelism_size > 1:
            # We need to offset the indices for the cache
            arg_affinities = {key + 4: arg_affinities[key] for key in arg_affinities}

            # Inputs have default affinity 0
            for i in range(4):
                arg_affinities[i] = DeviceAffinity("0")

        dynamic_shapes = {
            "tokens": {},
            "seq_lens": {},
            "start_positions": {},
            "seq_block_ids": {1: block_dim},
            "cache_state": cache_dynamic_shapes,
        }

        print(f"Exporting decode_bs{bs}")

        @fxb.export_program(
            name=f"decode_bs{bs}",
            args=(
                tokens,
                seq_lens,
                start_positions,
                seq_block_ids,
                cache_state,
            ),
            dynamic_shapes=dynamic_shapes,
            strict=args.strict,
            arg_device=arg_affinities,
        )
        def _(
            model,
            tokens,
            seq_lens,
            start_positions,
            seq_block_ids,
            cache_state,
        ):
            input_mask = model.input_mask(
                seq_lens, seq_block_ids.shape[1] * model.cache.block_seq_stride
            )
            attention_mask = model.decode_attention_mask(input_mask)

            if llama_config.tensor_parallelism_size != 1:
                shard_count = llama_config.tensor_parallelism_size

                tokens = ops.replicate(tokens, count=shard_count)
                attention_mask = ops.replicate(attention_mask, count=shard_count)
                start_positions = ops.replicate(start_positions, count=shard_count)
                seq_block_ids = ops.replicate(seq_block_ids, count=shard_count)

                cache_state = repack_cache(cache_state, cache_shard_dim)

            logits = model.decode(
                tokens,
                attention_mask=attention_mask,
                start_positions=start_positions,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )

            if llama_config.tensor_parallelism_size != 1:
                logits = ops.unshard(logits)

            return logits

    bsizes = []
    for bs in args.bs:
        if not args.skip_prefill:
            generate_batch_prefill(bs)
        if not args.skip_decode:
            generate_batch_decode(bs)
        bsizes.append(bs)
    config = generate_params_json(hp, bsizes, bsizes)
    print("GENERATED!")

    if args.verbose:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb, import_symbolic_shape_expressions=True)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)
    json.dump(config, open(args.output_config, "w"))


if __name__ == "__main__":
    main()
