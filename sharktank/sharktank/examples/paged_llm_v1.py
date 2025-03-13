# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inference support for the PagedLLMV1 protocol of models."""

import torch

# TODO: Should be using a base class with the protocol supported.
from ..models.mixtral.mixtral import *
from ..models.grok.grok import *
from ..models.llama.llama import *
from ..models.llama.sharding import shard_theta
from ..layers import *
from ..types import *
from sharktank.utils.load_llm import *
from sharktank.utils import cli

def main():

    parser = cli.create_parser()
    cli.add_input_dataset_options(parser)
    cli.add_tokenizer_options(parser)
    cli.add_quantization_options(parser)
    cli.add_model_options(parser)
    cli.add_model_input_options(parser)
    cli.add_save_tensor_options(parser)

    args = cli.parse(parser)
    device = torch.device(args.device) if args.device else None
    dataset = cli.get_input_dataset(args)
    tokenizer = cli.get_tokenizer(args)
    prompts = args.prompt
    
    config = LlamaModelConfig(
        hp=configs.LlamaHParams.from_gguf_props(dataset.properties),
        block_seq_stride=args.block_seq_stride,
        device=device,
        activation_dtype=args.activation_dtype,
        attention_dtype=args.attention_dtype,
        attention_kernel=args.attention_kernel,
        kv_cache_dtype=args.kv_cache_dtype,
        use_hf=args.use_hf,
        tensor_parallelism_size=args.tensor_parallelism_size,
        fake_quant=args.fake_quant,
    )
    if config.tensor_parallelism_size > 1:
        dataset.root_theta = shard_theta(dataset.root_theta, config)

    if config.hp.expert_count:
        if config.hp.model_arch == "grok":
            model = PagedGrokModelV1(dataset.root_theta, config)
        else:
            model = PagedMixtralModelV1(dataset.root_theta, config)
    else:
        model = PagedLlamaModelV1(dataset.root_theta, config)

    if args.save_intermediates_path:
        from ..utils.patching import SaveModuleResultTensorsPatch

        intermediates_saver = SaveModuleResultTensorsPatch()
        intermediates_saver.patch_child_modules(model)
    generator = TorchGenerator(model, tokenizer, dump_bins=args.dump_bins)

    print(f":: Prompting:")
    for prompt in prompts:
        print(f"    {prompt.encode()}")

    token_ids, seq_lens = generator.preprocess_input(prompts)
    batch = generator.begin_batch(token_ids=token_ids, seq_lens=seq_lens)
    
    print(f":: Prompt tokens: {batch.token_ids}")
    batch.prefill()
    print(f":: Prefill results:\n{batch.next_tokens.tolist()}")
    print(batch.detokenize())

    if args.save_intermediates_path:
        intermediates_saver.save_file(
            args.save_intermediates_path + "_prefill.safetensors"
        )
    if not args.skip_decode:
        counter = 0
        while not batch.done:
            batch.decode(batch.next_tokens)
            if args.save_intermediates_path:
                intermediates_saver.save_file(
                    args.save_intermediates_path + f"_step_{counter}.safetensors"
                )
            print(f":: Result tokens: {batch.results}")
            batch.print_current_results()
            counter += 1


if __name__ == "__main__":
    main()
