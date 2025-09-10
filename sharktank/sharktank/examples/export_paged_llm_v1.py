# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Export support for the PagedLLMV1 protocol of models."""

import dataclasses
import os
import logging
import json
import torch

from iree.turbine.aot import *
from sharktank.layers import BaseCausalLMModel
from sharktank.layers.configs import LlamaModelConfig, LlamaHParams, ParallelismConfig
from sharktank.layers.kv_cache import CacheAllocation
from sharktank.types import Theta
from sharktank.types.pipelining import pipeline_parallelize_llm_theta
from sharktank.utils import cli
from sharktank.utils.math import ceildiv
from sharktank.models.llm import PagedLlmModelV1
from sharktank.models.llm.config import ExportConfig
from sharktank.models.llm.export import ServicePagedLlmModelV1, build_service_config


def export_llm_v1(
    llama_config: LlamaModelConfig,
    theta: Theta,
    export_config: ExportConfig,
    strict: bool = False,
    loglevel: int = logging.DEBUG,
    modelClass: BaseCausalLMModel = PagedLlmModelV1,
):
    assert llama_config.tensor_parallelism_size == 1

    if export_config.top_k is not None and export_config.top_k < 1:
        raise NotImplementedError(f"`top-k` value must be >= 1.")

    model = modelClass(theta, llama_config)
    model = ServicePagedLlmModelV1(model=model, config=export_config)
    hp = llama_config.hp

    fxb = FxProgramsBuilder(model)

    def generate_batch_prefill(bs: int):
        # torch.export.Dim would make min at least 2
        block_dim_min = 2
        block_dim_max = ceildiv(hp.context_length, llama_config.block_seq_stride) - 1
        block_dim = torch.export.Dim("block", min=block_dim_min, max=block_dim_max)

        sl_dim = llama_config.block_seq_stride * block_dim

        start_pos = torch.empty(bs, dtype=torch.int64)
        cache, cache_dynamic_shapes, cache_affinities = model.setup_cache()

        dynamic_shapes = {
            "tokens": {1: sl_dim},
            "seq_lens": {},
            "seq_block_ids": {1: block_dim},
            "cs": cache_dynamic_shapes,
        }

        if export_config.use_extend_attention:
            bs_min = 2
            bs_max = ceildiv(llama_config.block_seq_stride, bs)
            extend_bs = torch.export.Dim("extend_bs", min=bs_min, max=bs_max)
            dynamic_shapes["tokens"][0] = extend_bs
            dynamic_shapes["seq_lens"][0] = extend_bs
            dynamic_shapes["seq_block_ids"][0] = extend_bs
        else:
            bs_min = bs

        seq_block_ids = torch.empty(bs_min, block_dim_min, dtype=torch.int64)

        tokens = torch.empty(
            bs_min,
            seq_block_ids.shape[1] * llama_config.block_seq_stride,
            dtype=torch.int64,
        )
        seq_lens = torch.empty(bs_min, dtype=torch.int64)

        print(f"Exporting prefill_bs{bs}")

        if export_config.has_prefill_position:
            dynamic_shapes["start_pos"] = {}
            arg_devices = model.setup_arg_devices(cache_affinities, len(dynamic_shapes))

            @fxb.export_program(
                name=f"prefill_bs{bs}",
                args=(tokens, start_pos, seq_lens, seq_block_ids, cache),
                dynamic_shapes=dynamic_shapes,
                arg_device=arg_devices,
                strict=strict,
            )
            def _(
                model: ServicePagedLlmModelV1,
                tokens,
                start_pos,
                seq_lens,
                seq_block_ids,
                cs,
            ):
                cache_state = CacheAllocation(allocation=cs)
                return model.prefill(
                    tokens, start_pos, seq_lens, seq_block_ids, cache_state
                )

        else:
            arg_devices = model.setup_arg_devices(cache_affinities, len(dynamic_shapes))

            @fxb.export_program(
                name=f"prefill_bs{bs}",
                args=(tokens, seq_lens, seq_block_ids, cache),
                dynamic_shapes=dynamic_shapes,
                arg_device=arg_devices,
                strict=strict,
            )
            def _(model: ServicePagedLlmModelV1, tokens, seq_lens, seq_block_ids, cs):
                cache_state = CacheAllocation(allocation=cs)
                start_pos = None
                return model.prefill(
                    tokens, start_pos, seq_lens, seq_block_ids, cache_state
                )

    def generate_batch_decode(bs: int):
        # torch.export.Dim would make min at least 2
        block_dim_min = 2
        block_dim_max = ceildiv(hp.context_length, llama_config.block_seq_stride) - 1
        block_dim = torch.export.Dim("block", min=block_dim_min, max=block_dim_max)

        tokens = torch.empty(bs, 1, dtype=torch.int64)
        seq_lens = torch.empty(bs, dtype=torch.int64)
        start_positions = torch.ones(bs, dtype=torch.int64)
        seq_block_ids = torch.empty(bs, block_dim_min, dtype=torch.int64)

        cache_state, cache_dynamic_shapes, cache_affinities = model.setup_cache()

        dynamic_shapes = {
            "tokens": {},
            "seq_lens": {},
            "start_positions": {},
            "seq_block_ids": {1: block_dim},
            "cache_state": cache_dynamic_shapes,
        }

        arg_devices = model.setup_arg_devices(cache_affinities, len(dynamic_shapes))

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
            arg_device=arg_devices,
            strict=strict,
        )
        def _(
            model: ServicePagedLlmModelV1,
            tokens,
            seq_lens,
            start_positions,
            seq_block_ids,
            cache_state,
        ):
            cache_state = CacheAllocation(allocation=cache_state)
            return model.decode(
                tokens,
                seq_lens,
                start_positions,
                seq_block_ids,
                cache_state,
            )

    if not export_config.skip_prefill:
        for bs in export_config.bs_prefill:
            generate_batch_prefill(bs)
    if not export_config.skip_decode:
        for bs in export_config.bs_decode:
            generate_batch_decode(bs)

    service_config = build_service_config(
        llama_config,
        export_config=export_config,
    )
    print("GENERATED!")

    if loglevel == logging.DEBUG:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb, import_symbolic_shape_expressions=True)

    return output, service_config


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

    dataset = cli.get_input_dataset(args)

    # Configure model export config from cli args:
    export_config = ExportConfig(
        top_k=args.top_k,
        device_block_count=args.device_block_count,
        logits_normalization=args.logits_normalization,
        prefill_final_logits=args.prefill_final_logits,
        use_linalgext_topk=args.use_linalgext_topk,
        has_prefill_position=args.has_prefill_position,
        use_extend_attention=args.use_extend_attention,
        bs_prefill=args.bs_prefill,
        bs_decode=args.bs_decode,
        skip_prefill=args.skip_prefill,
        skip_decode=args.skip_decode,
    )

    # Configure llama model form cli args:
    dtype_flags = cli.get_dtype_flags(args)
    llama_config = LlamaModelConfig.from_dataset(
        dataset=dataset,
        use_hf=args.use_hf,
        attention_kernel=args.attention_kernel,
        matmul_kernel=args.matmul_kernel,
        block_seq_stride=args.block_seq_stride,
        **dtype_flags,
    )

    hp = llama_config.hp
    parallelism_config = ParallelismConfig.default_config(
        block_count=hp.block_count,
        tp=args.tensor_parallelism_size,
        pp=args.pipeline_parallelism_size,
    )
    llama_config.parallelism_config = parallelism_config

    llama_config.fake_quant = args.fake_quant

    # These should be configured by the model source and not flags
    llama_config.use_qk_norm = args.use_qk_norm
    llama_config.attention_chunk_size = args.attention_chunk_size

    if "tensor_parallelism_size" in dataset.properties:
        if (
            dataset.properties["tensor_parallelism_size"]
            != llama_config.tensor_parallelism_size
        ):
            raise ValueError("Dataset tensor parallelism does not match flags")

    pipeline_parallelize_llm_theta(dataset.root_theta, llama_config.parallelism_config)

    output_export, output_config = export_llm_v1(
        llama_config=llama_config,
        theta=dataset.root_theta,
        export_config=export_config,
        strict=args.strict,
        loglevel=args.loglevel,
    )

    print(f"Saving to '{args.output_mlir}'")
    output_export.save_mlir(args.output_mlir)

    output_config = dataclasses.asdict(output_config)
    json.dump(output_config, open(args.output_config, "w"))


if __name__ == "__main__":
    main()
