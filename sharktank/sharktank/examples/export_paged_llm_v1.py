# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import os
import logging
import json
from typing import Any, Dict, Tuple
import torch

from iree.turbine.aot import *

from sharktank.layers import *
from sharktank.types import *
from sharktank.types.pipelining import pipeline_parallelize_theta
from sharktank.utils.math import ceildiv
from sharktank import ops
from sharktank.utils import cli

# TODO: Should be using a base class with the protocol supported.
from sharktank.models.llm import *


def main():
    parser = cli.create_parser()

    cli.add_input_dataset_options(parser)
    cli.add_model_options(parser)
    cli.add_export_artifacts(parser)
    cli.add_quantization_options(parser)
    cli.add_log_options(parser)

    args = cli.parse(parser)

    if args.output_mlir and args.output_mlir != "-":
        mlir_dir = os.path.dirname(args.output_mlir)
        if mlir_dir and not os.path.exists(mlir_dir):
            raise ValueError(
                f"Parent directory for output MLIR file does not exist: {mlir_dir}"
            )

    if args.top_k is not None and args.top_k < 1:
        raise NotImplementedError(f"`top-k` value must be >= 1.")

    if args.attention_kernel == "sharktank":
        ops.attention_impls.register_attention_override_by_name(
            "masked_flash_attention"
        )
    dataset_type = cli.get_input_data_files(args)
    dataset_type = "irpa" if "irpa" in dataset_type else "gguf"
    dataset = cli.get_input_dataset(args)
    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)
    if "tensor_parallelism_size" in dataset.properties:
        dataset_tensor_parallelism_size = dataset.properties["tensor_parallelism_size"]
        if dataset_tensor_parallelism_size != args.tensor_parallelism_size:
            raise ValueError(
                f"Tensor parallelism size mismatch: dataset={dataset_tensor_parallelism_size} while arg={args.tensor_parallelism_size}. Wrong value for --tensor-parallelism-size."
            )
    else:
        if args.tensor_parallelism_size != 1:
            raise ValueError(
                f"Unsharded dataset file provided, but specified --tensor-parallelism-size={args.tensor_parallelism_size}. Likely wrong dataset provided."
            )

    block_to_pipeline, pipeline_to_devices = pipeline_parallelize_theta(
        dataset.root_theta, args.pipeline_parallelism_size
    )

    llama_config = LlamaModelConfig(
        hp,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        block_to_pipeline_map=block_to_pipeline,
        pipeline_to_device_map=pipeline_to_devices,
        use_hf=args.use_hf,
        static_tables=False,  # Rely on the compiler for hoisting tables.
        attention_kernel=args.attention_kernel,
        block_seq_stride=args.block_seq_stride,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    llama_config.fake_quant = args.fake_quant

    model = PagedLlmModelV1(dataset.root_theta, llama_config)

    def generate_params_json(
        hp: LlamaHParams,
        prefill_bs: list[int],
        decode_bs: list[int],
        logits_normalization: str,
    ) -> Dict[str, Any]:
        """
        Generate config.json for shortfin.


        For shortfin, we only write attention_head_count_kv because that's all shortfin needs.
        Note that this is different from hp.attn_head_count when grouped attention shares kvcache between heads.
        """
        kv_cache_dtype = (
            str(llama_config.kv_cache_dtype).split(".")[-1]
            if llama_config.kv_cache_dtype is not None
            else str(llama_config.attention_dtype).split(".")[-1]
        )

        return {
            "module_name": "module",
            "module_abi_version": 1,
            "max_seq_len": hp.context_length,
            "attn_head_dim": hp.attn_head_dim,
            "prefill_batch_sizes": prefill_bs,
            "decode_batch_sizes": decode_bs,
            "transformer_block_count": hp.block_count,
            "logits_normalization": logits_normalization,
            "top_k": args.top_k,
            "paged_kv_cache": {
                "attention_head_count_kv": hp.attention_head_count_kv,
                "block_seq_stride": llama_config.block_seq_stride,
                "device_block_count": args.device_block_count,  # so that this makes its way into the config file & can be edited.
                "kv_cache_dtype": kv_cache_dtype,
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

            pipeline_parallelism_size = len(cache_state)
            tensor_parallelism_size = 1
            if isinstance(cache_state[0], ShardedTensor):
                tensor_parallelism_size = cache_state[0].shard_count
            parallelized = pipeline_parallelism_size > 1 or tensor_parallelism_size > 1

            dynamic_shapes = []
            for _ in range(pipeline_parallelism_size):
                ds = {0: page_dim}
                if parallelized:
                    ds = [ds] * tensor_parallelism_size
                dynamic_shapes.append(ds)
            unpacked = cache_state
            arg_affinities = {}
            shard_dim = None

            # Need to unpack that state when sharded (for tracing support reasons)
            if parallelized:
                shard_dim = cache_state[0].shard_dim

                unpacked = [[shard._data for shard in cs.shards] for cs in cache_state]

                # Cache is unpacked as [[pipeline 0 shards], [pipeline 1 shards], ...]
                # Therefore pipeline index is in outer loop.
                for pipeline, cache_state_for_pipeline in enumerate(cache_state):
                    for shard, device in enumerate(cache_state_for_pipeline.devices):
                        i = pipeline * tensor_parallelism_size + shard
                        arg_affinities[i] = DeviceAffinity(device)

            return unpacked, shard_dim, dynamic_shapes, arg_affinities
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

    def repack_cache(
        cache, shard_dim, pipeline_to_device_map: tuple[tuple[int, ...], ...]
    ) -> list[ShardedTensor]:
        return [
            (
                SplitPrimitiveTensor(
                    ts=c,
                    shard_dim=shard_dim,
                    devices=pipeline_to_device_map[pipeline],
                )
                if len(c) > 1
                else ReplicatedTensor(ts=c, devices=pipeline_to_device_map[pipeline])
            )
            for pipeline, c in enumerate(cache)
        ]

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

        if (
            llama_config.tensor_parallelism_size > 1
            or llama_config.pipeline_parallelism_size > 1
        ):
            # We need to offset the indices for the cache
            arg_affinities = {key + 3: arg_affinities[key] for key in arg_affinities}

            # Inputs have default affinity 0
            for i in range(3):
                device = pipeline_to_devices[0][0] if pipeline_to_devices else 0
                arg_affinities[i] = DeviceAffinity(device)

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
            cache_tensors = cs

            attention_mask = None
            if args.use_attention_mask:
                sl = tokens.shape[1]
                input_mask = model.input_mask(seq_lens, sl)
                attention_mask = model.attention_mask(input_mask)

            if (
                llama_config.tensor_parallelism_size == 1
                and llama_config.pipeline_parallelism_size == 1
            ):
                attention_mask = [attention_mask]
                seq_block_ids = [seq_block_ids]
            else:
                shard_count = llama_config.tensor_parallelism_size
                pipeline_to_device_map = (
                    llama_config.pipeline_to_device_map
                    if llama_config.pipeline_to_device_map
                    else [list(range(shard_count))]
                )

                tokens = ops.replicate(
                    tokens,
                    count=shard_count,
                    devices=pipeline_to_device_map[0],
                )
                if attention_mask is None:
                    attention_mask = [None] * len(pipeline_to_device_map)
                else:
                    attention_mask = [
                        ops.replicate(
                            attention_mask,
                            count=shard_count,
                            devices=pipeline_to_device_map[pipeline],
                        )
                        for pipeline in range(len(pipeline_to_device_map))
                    ]
                seq_block_ids = [
                    ops.replicate(
                        seq_block_ids,
                        count=shard_count,
                        devices=pipeline_to_device_map[pipeline],
                    )
                    for pipeline in range(len(pipeline_to_device_map))
                ]
                cache_tensors = repack_cache(
                    cs, cache_shard_dim, pipeline_to_device_map
                )

            logits = model.prefill(
                tokens,
                attention_mask=attention_mask,
                seq_block_ids=seq_block_ids,
                cache_state=cache_tensors,
            )

            if llama_config.tensor_parallelism_size != 1:
                logits = ops.unshard(logits)

            if args.logits_normalization == "softmax":
                logits = ops.softmax(logits, dim=-1)

            if args.logits_normalization == "log_softmax":
                logits = ops.elementwise(torch.log, ops.softmax(logits, dim=-1))

            top_k = args.top_k
            if top_k is None:
                return logits

            if top_k == 1:
                return argmax_output(logits, chunk_size=hp.context_length // 128)

            return topk_output(
                logits,
                k=args.top_k,
                chunk_size=256,
                use_linalgext_topk=args.use_linalgext_topk,
            )

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

        if (
            llama_config.tensor_parallelism_size > 1
            or llama_config.pipeline_parallelism_size > 1
        ):
            # We need to offset the indices for the cache
            arg_affinities = {key + 4: arg_affinities[key] for key in arg_affinities}

            # Inputs have default affinity 0
            for i in range(4):
                device = pipeline_to_devices[0][0] if pipeline_to_devices else 0
                arg_affinities[i] = DeviceAffinity(device)

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

            if (
                llama_config.tensor_parallelism_size == 1
                and llama_config.pipeline_parallelism_size == 1
            ):
                attention_mask = [attention_mask]
                seq_block_ids = [seq_block_ids]
                start_positions = [start_positions]
            else:
                shard_count = llama_config.tensor_parallelism_size
                pipeline_to_device_map = (
                    llama_config.pipeline_to_device_map
                    if llama_config.pipeline_to_device_map
                    else [list(range(shard_count))]
                )

                tokens = ops.replicate(
                    tokens,
                    count=shard_count,
                    devices=pipeline_to_device_map[0],
                )
                _attention_mask, _start_positions, _seq_block_ids = [], [], []
                for pipeline in range(len(pipeline_to_device_map)):
                    devices = pipeline_to_device_map[pipeline]
                    _attention_mask.append(
                        ops.replicate(
                            attention_mask, count=shard_count, devices=devices
                        )
                    )
                    _start_positions.append(
                        ops.replicate(
                            start_positions, count=shard_count, devices=devices
                        )
                    )
                    _seq_block_ids.append(
                        ops.replicate(seq_block_ids, count=shard_count, devices=devices)
                    )
                attention_mask, start_positions, seq_block_ids = (
                    _attention_mask,
                    _start_positions,
                    _seq_block_ids,
                )

                cache_state = repack_cache(
                    cache_state, cache_shard_dim, pipeline_to_device_map
                )

            logits = model.decode(
                tokens,
                attention_mask=attention_mask,
                start_positions=start_positions,
                seq_block_ids=seq_block_ids,
                cache_state=cache_state,
            )

            if llama_config.tensor_parallelism_size != 1:
                logits = ops.unshard(logits)

            if args.logits_normalization == "softmax":
                logits = ops.softmax(logits, dim=-1)

            if args.logits_normalization == "log_softmax":
                logits = ops.elementwise(torch.log, ops.softmax(logits, dim=-1))

            top_k = args.top_k
            if top_k is None:
                return logits

            if top_k == 1:
                return argmax_output(logits, chunk_size=hp.context_length // 128)

            return topk_output(
                logits,
                k=top_k,
                chunk_size=256,
                use_linalgext_topk=args.use_linalgext_topk,
            )

    def argmax_output(
        logits: torch.Tensor, chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the max logits and indices for the given logits.

        Args:
            logits (torch.Tensor): Logits tensor to find the max from.
            chunk_size (int): Chunk size for the argmax operation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the max logits and their indices.
        """
        indices = ops.argmax(logits, -1, chunk_size=chunk_size)
        indices_expanded = indices.unsqueeze(-1)

        max_logits = ops.gather(logits, dim=-1, index=indices_expanded)
        max_logits = max_logits.squeeze(-1)

        return max_logits, indices

    def topk_output(
        logits: torch.Tensor, k: int, chunk_size: int, use_linalgext_topk: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the top-k logits and their indices for the given logits.

        Args:
            logits (torch.Tensor): Logits tensor to find the top-k from.
            k (int): Number of top elements to return.
            chunk_size (int): Chunk size for the top-k operation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the top-k logits and their indices.
        """
        return ops.topk(
            logits,
            k=k,
            dim=-1,
            largest=True,
            sorted=not use_linalgext_topk,
            chunk_size=chunk_size,
            use_linalgext_topk=use_linalgext_topk,
        )

    if not args.skip_prefill:
        for bs in args.bs_prefill:
            generate_batch_prefill(bs)
    if not args.skip_decode:
        for bs in args.bs_decode:
            generate_batch_decode(bs)

    config = generate_params_json(
        hp, args.bs_prefill, args.bs_decode, args.logits_normalization
    )
    print("GENERATED!")

    if args.loglevel == logging.DEBUG:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb, import_symbolic_shape_expressions=True)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)
    json.dump(config, open(args.output_config, "w"))


if __name__ == "__main__":
    main()
