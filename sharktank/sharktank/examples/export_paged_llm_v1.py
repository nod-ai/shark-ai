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
from sharktank.layers.configs import LlamaModelConfig, LlamaHParams
from sharktank.layers.paged_attention import CacheAllocation
from sharktank.types import Theta
from sharktank.utils import cli
<<<<<<< HEAD
from sharktank.utils.math import ceildiv
from sharktank.models.llm import PagedLlmModelV1
from sharktank.models.llm.config import ExportConfig
from sharktank.models.llm.export import ServicePagedLlmModelV1, build_service_config
=======

# TODO: Should be using a base class with the protocol supported.
from sharktank.models.llm import *
from sharktank.models.llm.woof import *
>>>>>>> c4bd6dc32 (woofwip local)


def export_llm_v1(
    llama_config: LlamaModelConfig,
    theta: Theta,
    export_config: ExportConfig,
    strict: bool = False,
    loglevel: int = logging.DEBUG,
):
    assert llama_config.pipeline_parallelism_size == 1
    assert llama_config.tensor_parallelism_size == 1

    if export_config.top_k is not None and export_config.top_k < 1:
        raise NotImplementedError(f"`top-k` value must be >= 1.")

<<<<<<< HEAD
    model = PagedLlmModelV1(theta, llama_config)
    model = ServicePagedLlmModelV1(model=model, config=export_config)
    hp = llama_config.hp
=======
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

    model = PagedWoofModelV1(dataset.root_theta, llama_config)

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
>>>>>>> c4bd6dc32 (woofwip local)

    fxb = FxProgramsBuilder(model)

    def setup_cache(
        model: ServicePagedLlmModelV1,
    ) -> tuple[list[torch.Tensor], list[dict[int, torch.export.Dim]]]:
        if not model.is_paged:
            raise NotImplementedError(f"Unsupported KV cache type")

        device_block_count = export_config.device_block_count
        cache_state = model.allocate_cache(page_count=device_block_count)
        page_dim = torch.export.Dim("page")

        unpacked = cache_state.allocation
        dynamic_shapes = [{0: page_dim}]

        return unpacked, dynamic_shapes

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
        start_pos = torch.empty(bs, dtype=torch.int64)
        seq_lens = torch.empty(bs, dtype=torch.int64)

        cache, cache_dynamic_shapes = setup_cache(model)

        dynamic_shapes = {
            "tokens": {1: sl_dim},
            "seq_lens": {},
            "seq_block_ids": {1: block_dim},
            "cs": cache_dynamic_shapes,
        }

        print(f"Exporting prefill_bs{bs}")

        if export_config.has_prefill_position:
            dynamic_shapes["start_pos"] = {}

            @fxb.export_program(
                name=f"prefill_bs{bs}",
                args=(tokens, start_pos, seq_lens, seq_block_ids, cache),
                dynamic_shapes=dynamic_shapes,
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

            @fxb.export_program(
                name=f"prefill_bs{bs}",
                args=(tokens, seq_lens, seq_block_ids, cache),
                dynamic_shapes=dynamic_shapes,
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
        use_attention_mask=args.use_attention_mask,
        use_linalgext_topk=args.use_linalgext_topk,
        has_prefill_position=args.has_prefill_position,
        bs_prefill=args.bs_prefill,
        bs_decode=args.bs_decode,
        skip_prefill=args.skip_prefill,
        skip_decode=args.skip_decode,
    )

    # Configure llama model form cli args:
    hp = LlamaHParams.from_gguf_props(dataset.properties)
    llama_config = LlamaModelConfig(
        hp,
        tensor_parallelism_size=args.tensor_parallelism_size,
        pipeline_parallelism_size=args.pipeline_parallelism_size,
        use_hf=args.use_hf,
        static_tables=False,  # Rely on the compiler for hoisting tables.
        attention_kernel=args.attention_kernel,
        matmul_kernel=args.matmul_kernel,
        block_seq_stride=args.block_seq_stride,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
    )

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
