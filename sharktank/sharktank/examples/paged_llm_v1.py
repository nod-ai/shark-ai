# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inference support for the PagedLLMV1 protocol of models."""

import logging
import torch

# TODO: Should be using a base class with the protocol supported.
from amdsharktank.layers.configs.llm_configs import ParallelismConfig
from amdsharktank.types.sharding import shard_theta
from amdsharktank.types.tensors import dtype_to_serialized_name
from amdsharktank.layers import LlamaModelConfig
from amdsharktank.utils.llm_utils import (
    TorchInstance,
    LlmInstance,
    llama_config_page_sizes,
)
from amdsharktank.types.pipelining import pipeline_parallelize_llm_theta
from amdsharktank.utils import cli


def main(cli_args: list[str] | None = None):
    """
    Run LLM inference in torch/eager mode. Use --device='cuda:0' to run on AMD GPU
    Args:
        --prompt: list[str] - Custom space separated prompts
        --prompt-seq-len: int - Generate random token ids for given seq len and bs and save prefill & first decode step input args as npy files
        --dump-path: str - Path to save prefill and decode input args as npy files
        --dump-decode-steps: int - Number of decode steps to dump decode args (defaults to 1 decode step)
        --max-decode-steps: int - maximum number of decode steps to perform.
        --bs: int - batch size, for custom prompts, bs is number of given prompts (defaults to 4)
        --save_intermediates_path: str - save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors"
    """
    from ..utils import cli

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    cli.add_quantization_options(parser)
    cli.add_model_options(parser)
    cli.add_model_input_options(parser)
    cli.add_save_tensor_options(parser)

    args = cli.parse(parser, args=cli_args)

    device = torch.device(args.device) if args.device else None
    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)
    dtype_flags = cli.get_dtype_flags(args)

    config = LlamaModelConfig.from_dataset(
        dataset=dataset,
        block_seq_stride=args.block_seq_stride,
        device=device,
        attention_kernel=args.attention_kernel,
        matmul_kernel=args.matmul_kernel,
        fake_quant=args.fake_quant,
        **dtype_flags,
    )

    config.parallelism_config = ParallelismConfig.default_config(
        block_count=config.hp.block_count,
        pp=args.pipeline_parallelism_size,
        tp=args.tensor_parallelism_size,
    )

    pipeline_parallelize_llm_theta(dataset.root_theta, config.parallelism_config)

    if args.use_hf:
        logging.log(logging.WARNING, "Use HF will be deprecated 10/01/2025")
        config.hp.rope_interleave_emb = False

    if args.tensor_parallelism_size != config.tensor_parallelism_size:
        assert (
            config.tensor_parallelism_size == 1
        ), "Can't tensor-shard theta that is already sharded"
        config.tensor_parallelism_size = args.tensor_parallelism_size
        dataset.root_theta = shard_theta(dataset.root_theta, config)

    model = TorchInstance(
        theta=dataset.root_theta,
        config=config,
        device=device,
        prefill_bs=args.bs,
        decode_bs=args.bs,
    )

    if args.save_intermediates_path:
        from amdsharktank.utils.patching import SaveModuleResultTensorsPatch

        intermediates_saver = SaveModuleResultTensorsPatch()
        intermediates_saver.patch_child_modules(model._model)

    page_sizes = llama_config_page_sizes(model.config)

    # TODO: block_count should be config.hp.block_count,
    # but currently pages are not being used efficiently,
    # which is causing memory issues with lower number of pages.
    # So, keeping at least 8 pages for now.
    new_block_count = max(config.hp.block_count, 8)
    llm_instance = LlmInstance(
        model_instance=model,
        page_sizes=page_sizes,
        block_seq_stride=args.block_seq_stride,
        block_count=new_block_count,
        kv_cache_dtype=dtype_to_serialized_name(model._model.cache.cache_dtype),
    )

    decoder = llm_instance.make_decoder()

    assert (args.prompt is None) ^ (
        args.prompt_seq_len is None
    ), 'Exactly one of "--prompt" or "--prompt-seq-len" must be provided'

    if args.prompt_seq_len is not None:
        torch.random.manual_seed(0)
        token_ids = torch.randint(
            low=0,
            high=int(model._model.config.hp.vocab_size),
            size=(args.bs, args.prompt_seq_len),
            device=model._model.device,
        ).tolist()
    else:
        token_ids = tokenizer.encode(texts=args.prompt, add_start_token=False)[0]

    results = decoder.greedy_decode(token_ids, args.max_decode_steps)
    print(f":: Result tokens: {results}")
    print(f":: Result: {tokenizer.decode(results)}")


if __name__ == "__main__":
    main()
