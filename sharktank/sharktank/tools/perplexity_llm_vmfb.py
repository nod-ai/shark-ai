# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import dataclasses
import json
import tokenizers

from datasets import load_dataset

from sharktank.models.llm.config import ServiceConfig
from sharktank.utils.llm_utils import IreeInstance, LlmInstance


class Tokenizer:
    def __init__(self, fp):
        self.t = tokenizers.Tokenizer.from_file(fp)

    def encode(self, texts: list[str]) -> list[list[int]]:
        """Encodes a batch of texts, applying no padding."""
        return [s.ids for s in self.t.encode_batch(texts)]

    def decode(self, sequences) -> list[str]:
        """Decodes a batch of sequences to text."""
        return self.t.decode_batch(sequences)


def main(dataset, vmfb, config, irpa, tokenizer):
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

    iree = IreeInstance(devices=["hip://0"], vmfb=vmfb, parameters=irpa)
    server_config = ServiceConfig.load(config)
    llm = LlmInstance.load(iree, server_config)

    runner = llm.make_perplexity_eval()
    results = runner.batch_prefill_perplexity(requests=encoded)

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
    parser.add_argument("--dataset", help="Path to dataset", required=True)
    parser.add_argument("--irpa", help="IRPA parameters file", required=True)
    parser.add_argument("--vmfb", help="vmfb file path", required=True)
    parser.add_argument("--config", help="json config file for server", required=True)
    parser.add_argument("--tokenizer", help="json tokenizer config file", required=True)
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        irpa=args.irpa,
        vmfb=args.vmfb,
        config=args.config,
        tokenizer=args.tokenizer,
    )
