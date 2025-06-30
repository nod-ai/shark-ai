# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inference support for the PagedLLMV1 protocol of models."""

import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Should be using a base class with the protocol supported.
from sharktank.models.llm import *
from sharktank.types.sharding import shard_theta
from sharktank.layers import *
from sharktank.types import *
from sharktank.utils.load_llm import *
from sharktank.utils import cli


def main(cli_args: list[str] | None = None):
    """
    Run LLM inference in torch/eager mode. Use --device='cuda:0' to run on AMD GPU
    Args:
        --prompt: list[str] - Custom space separated prompts
        --prompt-seq-len: int - Generate random token ids for given seq len and bs and save prefill & first decode step input args as npy files
        --dump-path: str - Path to save prefill and decode input args as npy files
        --dump-decode-steps: int - Number of decode steps to dump decode args (defaults to 1 decode step)
        --bs: int - batch size, for custom prompts, bs is number of given prompts (defaults to 4)
        --save_intermediates_path: str - save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors"
    """
    from ..utils import cli

    with torch.no_grad():
        parser = cli.create_parser()
        cli.add_input_dataset_options(parser)
        cli.add_tokenizer_options(parser)
        cli.add_quantization_options(parser)
        cli.add_model_options(parser)
        cli.add_model_input_options(parser)
        cli.add_save_tensor_options(parser)

        args = cli.parse(parser, args=cli_args)

        device = torch.device(args.device) if args.device else None
        logger.info("Loading dataset ...")
        dataset = cli.get_input_dataset(args)
        logger.info("Dataset loaded.")
        tokenizer = cli.get_tokenizer(args)

        config = LlamaModelConfig.from_properties(dataset.properties)
        config.block_seq_stride = args.block_seq_stride
        config.device = device
        config.activation_dtype = args.activation_dtype
        config.attention_dtype = args.attention_dtype
        config.attention_kernel = args.attention_kernel
        config.kv_cache_dtype = args.kv_cache_dtype
        config.use_hf = args.use_hf
        config.pipeline_parallelism_size = args.pipeline_parallelism_size
        config.fake_quant = args.fake_quant

        # import gc
        # import sharktank.ops

        # # import sharktank.models.llama.testing
        # # sharktank.models.llama.testing.make_random_llama_theta(config=config, vocab_size=config.hp.vocab_size)
        # logger.info("Conversion from f16 to bf16 ...")

        # def convert_f16_to_bf16(tensor: InferenceTensor) -> InferenceTensor:
        #     if tensor.dtype == torch.float16:
        #         # return tensor.to(dtype=torch.bfloat16)
        #         return sharktank.ops.zeros_like(tensor, dtype=torch.bfloat16)
        #     return tensor

        # flat_theta = dataset.root_theta.flatten()
        # dataset.root_theta = None
        # flat_theta_converted = {}
        # while len(flat_theta) > 0:
        #     for name in flat_theta.keys():
        #         flat_theta_converted[name] = convert_f16_to_bf16(flat_theta[name])
        #         del flat_theta[name]
        #         gc.collect()
        #         break
        # del flat_theta
        # gc.collect()
        # # dataset.root_theta = Theta(
        # #     {
        # #         name: convert_f16_to_bf16(tensor) for name, tensor in dataset.root_theta.flatten().items()
        # #     }
        # # )
        # return

        # config.activation_dtype = torch.bfloat16
        # config.attention_dtype = torch.bfloat16
        # config.kv_cache_dtype = torch.bfloat16
        # logger.info("Conversion from f16 to bf16 done.")
        # dtypes = set([tensor.dtype for tensor in dataset.root_theta.flatten().values()])
        # logger.info(f"Dtypes {dtypes}.")

        if args.tensor_parallelism_size != config.tensor_parallelism_size:
            assert (
                config.tensor_parallelism_size == 1
            ), "Can't tensor-shard theta that is already sharded"
            config.tensor_parallelism_size = args.tensor_parallelism_size
            dataset.root_theta = shard_theta(dataset.root_theta, config)

        model = PagedLlmModelV1(dataset.root_theta, config)

        if args.save_intermediates_path:
            from sharktank.utils.patching import SaveModuleResultTensorsPatch

            intermediates_saver = SaveModuleResultTensorsPatch()
            intermediates_saver.patch_child_modules(model)

        generator = TorchGenerator(
            model, tokenizer, max_decode_steps=args.max_decode_steps
        )

        assert (
            args.prompt is None or args.prompt_seq_len is None
        ), 'CLI args "--prompt-seq-len" and "--prompt" are mutually exclusive'
        assert (
            args.prompt is not None or args.prompt_seq_len is not None
        ), 'Exactly one of CLI args "--prompt-seq-len" and "--prompt" must be provided.'
        if args.prompt_seq_len is not None:
            torch.random.manual_seed(0)
            token_ids, seq_lens = generator.generate_random_tokens(
                batch_size=args.bs, prompt_seq_len=args.prompt_seq_len
            )
        else:
            token_ids, seq_lens = generator.preprocess_prompts(prompts=args.prompt)
        batch = generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
            dump_path=args.dump_path,
            dump_decode_steps=args.dump_decode_steps,
            use_attention_mask=args.use_attention_mask,
        )
        logger.info("Prefill ...")
        results = batch.prefill()
        logger.info("Prefill done.")
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
