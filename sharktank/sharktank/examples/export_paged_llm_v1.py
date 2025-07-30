# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import os
import logging
import json
from typing import Any, Dict, Tuple, Optional
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

    dataset_type = cli.get_input_data_files(args)
    dataset_type = "irpa" if "irpa" in dataset_type else "gguf"
    dataset = cli.get_input_dataset(args)
    hp = configs.LlamaHParams.from_gguf_props(dataset.properties)

    assert args.pipeline_parallelism_size == 1
    assert args.tensor_parallelism_size == 1

    if "tensor_parallelism_size" in dataset.properties:
        if (
            dataset.properties["tensor_parallelism_size"]
            != args.tensor_parallelism_size
        ):
            raise ValueError("Dataset tensor parallelism does not match flags")

    llama_config = LlamaModelConfig(
        hp,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        use_hf=args.use_hf,
        static_tables=False,  # Rely on the compiler for hoisting tables.
        attention_kernel=args.attention_kernel,
        block_seq_stride=args.block_seq_stride,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    llama_config.fake_quant = args.fake_quant
    llama_config.use_qk_norm = args.use_qk_norm
    llama_config.attention_chunk_size = args.attention_chunk_size

    model = PagedLlmModelV1(dataset.root_theta, llama_config)

    def generate_params_json(
        llama_config: LlamaModelConfig,
        prefill_bs: list[int],
        decode_bs: list[int],
        logits_normalization: str,
    ) -> Dict[str, Any]:
        """
        Generate config.json for shortfin.


        For shortfin, we only write attention_head_count_kv because that's all shortfin needs.
        Note that this is different from hp.attn_head_count when grouped attention shares kvcache between heads.
        """
        hp = llama_config.hp

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
                # The compiler assumes that the page_dim cannot be greater
                # than the device block count. Be careful while modifying
                # this. Ideally, we want to allocate the number of pages such
                # that (head_dim * block_seq_stride * num_pages) <= int32_max,
                # to allow doing int32 indexing for kv cache gather/scatter,
                # which is good for buffer loads on gfx94x+.
                "device_block_count": args.device_block_count,
                "kv_cache_dtype": kv_cache_dtype,
            },
        }

    fxb = FxProgramsBuilder(model)

    def setup_cache(model):
        if model.config.kv_cache_type == "paged":
            cache_state = model.cache.allocate(page_count=args.device_block_count)
            page_dim = torch.export.Dim("page", max=args.device_block_count)

            unpacked = cache_state
            dynamic_shapes = [{0: page_dim}]

            return unpacked, dynamic_shapes
        else:
            raise NotImplementedError(f"Unsupported KV cache type: {type(model.cache)}")

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

        cache, cache_dynamic_shapes = setup_cache(model)

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
        )
        def _(model, tokens, seq_lens, seq_block_ids, cs):
            cache_tensors = cs

            attention_mask = None
            if args.use_attention_mask:
                sl = tokens.shape[1]
                input_mask = model.input_mask(seq_lens, sl)
                attention_mask = model.attention_mask(input_mask)

            attention_mask = attention_mask
            seq_block_ids = seq_block_ids

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

            if args.prefill_final_logits:
                last_seq_lens = seq_lens
                bsi = torch.tensor(list(range(logits.shape[0])))

                logits = logits[bsi, last_seq_lens - 1]
                logits = logits.unsqueeze(1)

            top_k = args.top_k
            if top_k is None:
                return logits

            if top_k == 1:
                return argmax_output(logits, chunk_size=None)

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

        cache_state, cache_dynamic_shapes = setup_cache(model)

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
                return argmax_output(logits, chunk_size=None)

            return topk_output(
                logits,
                k=top_k,
                chunk_size=256,
                use_linalgext_topk=args.use_linalgext_topk,
            )

    def argmax_output(
        logits: torch.Tensor, chunk_size: Optional[int]
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
        llama_config,
        args.bs_prefill,
        args.bs_decode,
        args.logits_normalization,
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
