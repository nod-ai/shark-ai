# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datasets import load_dataset
from pathlib import Path
from transformers.models import LlamaForCausalLM, AutoTokenizer
from sharktank.utils.llm_utils import LlmPerplexityEval
from sharktank.utils.tokenizer import load_tokenizer
import json
import safetensors.torch
import torch
import transformers


def generate_prefill_logits(
    model: LlamaForCausalLM, prompt_token_ids: list[list[int]]
) -> list[torch.Tensor]:
    res: list[torch.Tensor] = []
    for ids in prompt_token_ids:
        input_ids = torch.asarray(ids, device=model.device).unsqueeze(0)
        model_output = model(input_ids=input_ids)
        assert model_output.logits.shape[0] == 1
        res.append(model_output.logits.squeeze(0))
    return res


def save_logits(logits: list[torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logits = [
        torch.nn.functional.softmax(l, dim=-1, dtype=torch.float32) for l in logits
    ]
    logits = [l[:-1] for l in logits]
    concatenated_logits = torch.cat(logits)
    safetensors.torch.save_file(
        {"logits": concatenated_logits}, filename=f"{output_path}"
    )


def main():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    revision = "0e9e39f249a16976918f6564b8830bc894c89659"
    dataset = "/home/bpetkant/ws/sharktank/repo/sharktank/tests/evaluate/datasets/llama_8b_fp16_torch.json"
    output_path = "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/hf-logits/logits.safetensors"

    with open(dataset, "r") as dataset:
        dataset = LlmPerplexityEval.Dataset(**json.load(dataset))
    test_prompts = load_dataset(dataset.dataset, dataset.revision, split=dataset.split)[
        "text"
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, legacy=False)
    test_prompts = [test_prompts[id] for id in dataset.ids]
    prompt_token_ids = tokenizer.batch_encode_plus(
        test_prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    ).input_ids
    print(f"len(test_prompts[-1]) = {len(test_prompts[-1])}")
    print(f"len(prompt_token_ids[-1]) = {len(prompt_token_ids[-1])}")

    tokenizer = load_tokenizer(
        "/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    )
    encoded, lens = tokenizer.encode(test_prompts)
    encoded = [ids[:len] for ids, len in zip(encoded, lens)]
    print(f"len(test_prompts[-1]) = {len(test_prompts[-1])}")
    print(f"len(encoded[-1]) = {len(encoded[-1])}")
    assert False, "TODO: remove"

    model = LlamaForCausalLM.from_pretrained(
        model_id, revision=revision, device_map="auto", torch_dtype=torch.bfloat16
    )

    logits = generate_prefill_logits(model=model, prompt_token_ids=prompt_token_ids)
    save_logits(logits=logits, output_path=Path(output_path))


if __name__ == "__main__":
    main()

##########################################################################

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
revision = "0e9e39f249a16976918f6564b8830bc894c89659"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    revision=revision,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

# model = LlamaForCausalLM.from_pretrained(model_id, revision=revision, device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# prompt = "Hey, are you conscious? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt")
# input_ids = inputs.input_ids.to(device=model.device)

# prefill_output = model(input_ids=input_ids)
# pass

# #generate_ids = model.generate(input_ids, max_length=30)
# #answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# #print(answer)
