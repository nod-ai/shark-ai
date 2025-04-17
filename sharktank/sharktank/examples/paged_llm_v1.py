# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inference support for the PagedLLMV1 protocol of models."""

import torch

# TODO: Should be using a base class with the protocol supported.
from sharktank.models.llm import *
from sharktank.models.llama.sharding import shard_theta
from sharktank.layers import *
from sharktank.types import *
from sharktank.utils.load_llm import *
from sharktank.utils import cli


def main():
    from ..utils import cli

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

    model = PagedLlmModelV1(dataset.root_theta, config)

    if args.save_intermediates_path:
        from sharktank.utils.patching import SaveModuleResultTensorsPatch

        intermediates_saver = SaveModuleResultTensorsPatch()
        intermediates_saver.patch_child_modules(model)

    generator = TorchGenerator(model, tokenizer)

    token_ids, seq_lens = generator.preprocess_prompts(prompts=args.prompt)
    batch = generator.begin_batch(
        token_ids=token_ids,
        seq_lens=seq_lens,
        dump_bins=args.dump_bins,
    )
    results = batch.prefill()
    batch.print_current_results()

    if args.save_intermediates_path:
        intermediates_saver.save_file(
            args.save_intermediates_path + "_prefill.safetensors"
        )
    if not args.skip_decode:
        counter = 0
        while not batch.done:
            results = batch.decode(results)
            batch.print_current_results()

            if args.save_intermediates_path:
                intermediates_saver.save_file(
                    args.save_intermediates_path + f"_step_{counter}.safetensors"
                )
            print(f":: Result tokens: {batch.results}")
            batch.print_current_results()
            counter += 1

        if len(batch.parent.free_pages) == 0:
            print(
                "\n\n:: Out of allocated pages, increase page_cache_size to continue generation.\n"
            )


if __name__ == "__main__":
    main()
