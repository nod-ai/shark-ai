# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from pathlib import Path

import torch
from amdsharktank.layers.configs.llm_configs import *
from amdsharktank.examples.paged_llm_v1 import *
from amdsharktank.models.llm import *
from amdsharktank.utils import tokenizer, hf_datasets


class BaseLlamaTest(unittest.TestCase):
    def setUp(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def createConfigModel(self, kv_cache_type):
        return LlamaModelConfig.from_dataset(
            dataset=self.dataset,
            block_seq_stride=16,
            kv_cache_type=kv_cache_type,
            device=self.device,
            activation_dtype=self.activation_dtype,
            attention_dtype=self.activation_dtype,
        )

    def runPrefill(self, *, kv_cache_type):
        config = self.createConfigModel(kv_cache_type)
        model = PagedLlmModelV1(self.dataset.root_theta, config)
        generator = TorchGenerator(model, self.tokenizer_config)
        token_ids, seq_lens = generator.preprocess_prompts(prompts=self.prompts)
        batch = generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )

        results = batch.prefill()
        logits = batch.prefill_logits

        bs, *_ = logits.shape
        assert len(batch.seq_lens) == bs
        greedy_token_logit = 0.0
        step_logits = logits[0, batch.seq_lens[0] - 1]
        greedy_token_logit = step_logits[torch.argmax(step_logits)]

        return results, greedy_token_logit

    def comparePrefillResults(
        self,
        batch_results,
        greedy_token_logit,
        golden_prefill_token,
        golden_prefill_token_logit,
    ):
        rtol = 3e-4
        atol = 4e-3
        assert batch_results == golden_prefill_token
        torch.testing.assert_close(
            greedy_token_logit, golden_prefill_token_logit, rtol=rtol, atol=atol
        )


class Llama8BTest(BaseLlamaTest):
    def setUp(self):
        default_arguments = {
            "hf_dataset": "llama3_8B_f16",
            "tokenizer-config-json": Path("./llama3.1-8b/tokenizer_config.json"),
            "prompt": ["I believe the meaning of life is"],
            "device": None,
            "activation-dtype": "float32",
        }
        self.device = (
            torch.device(default_arguments["device"])
            if default_arguments["device"]
            else None
        )
        self.activation_dtype = getattr(torch, default_arguments["activation-dtype"])
        assert isinstance(self.activation_dtype, torch.dtype)
        self.data_files = hf_datasets.get_dataset(
            default_arguments["hf_dataset"]
        ).download(local_dir=Path("."))
        self.dataset = Dataset.load(self.data_files["gguf"][0], file_type="gguf")
        self.tokenizer_config = tokenizer.load_tokenizer(
            default_arguments["tokenizer-config-json"].parent,
            tokenizer_type="transformers",
        )
        self.prompts = default_arguments["prompt"]
        # token and logit determined by running llama.cpp (llama_cpp_instructions.md).
        self.llama_cpp_8b_prefill_token = [[311]]
        self.llama_cpp_8b_prefill_token_logit = torch.tensor(15.612568)

    def testPrefillPaged8B(self):
        batch_results_paged, greedy_token_logit_paged = self.runPrefill(
            kv_cache_type="paged"
        )
        self.comparePrefillResults(
            batch_results_paged,
            greedy_token_logit_paged,
            self.llama_cpp_8b_prefill_token,
            self.llama_cpp_8b_prefill_token_logit,
        )


if __name__ == "__main__":
    unittest.main()
