# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import tokenizers
import torch

from datasets import load_dataset

from sharktank.utils.llm_utils import TorchInstance, LlmInstance, llama_config_page_size


class Tokenizer:
    def __init__(self, fp):
        self.t = tokenizers.Tokenizer.from_file(fp)

    def encode(self, texts: list[str]) -> list[list[int]]:
        """Encodes a batch of texts, applying no padding."""
        return [s.ids for s in self.t.encode_batch(texts)]

    def decode(self, sequences) -> list[str]:
        """Decodes a batch of sequences to text."""
        return self.t.decode_batch(sequences)


def main(device, dataset, irpa, tokenizer, cross_entropy):
    torch.set_default_device(device)
    tokenizer = Tokenizer(tokenizer)

    with open(dataset, "r") as dataset:
        dataset = json.load(dataset)

    name = dataset["dataset"]
    revision = dataset["revision"]
    split = dataset["split"]
    ids = dataset["ids"]

    test_prompts = load_dataset(name, revision, split=split)["text"]
    test_prompts = [test_prompts[id] for id in ids]
    encoded = tokenizer.encode(test_prompts)

    torch_instance = TorchInstance.load(irpa, device=device)

    page_size = llama_config_page_size(torch_instance.config)
    block_count = 512

    llm = LlmInstance(
        model_instance=torch_instance,
        page_size=page_size,
        block_seq_stride=torch_instance.config.block_seq_stride,
        block_count=block_count,
    )

    runner = llm.make_perplexity_eval()
    results = runner.batch_prefill_perplexity(
        requests=encoded, cross_entropy=cross_entropy
    )

    scores = {id: result.score for id, result in zip(ids, results)}
    results = {
        "dataset": name,
        "revision": revision,
        "split": split,
        "scores": scores,
        "ids": ids,
    }

    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", help="Torch device to use for computation", default="cuda"
    )
    parser.add_argument("--dataset", help="Path to dataset", required=True)
    parser.add_argument("--irpa", help="IRPA parameters file", required=True)
    parser.add_argument("--tokenizer", help="json tokenizer config file", required=True)
    parser.add_argument(
        "--cross-entropy", help="return cross entropy value", action="store_true"
    )
    args = parser.parse_args()
    main(
        device=args.device,
        dataset=args.dataset,
        irpa=args.irpa,
        tokenizer=args.tokenizer,
        cross_entropy=args.cross_entropy,
    )
