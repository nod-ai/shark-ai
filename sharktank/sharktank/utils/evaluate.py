# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import random
import math
from datasets import load_dataset

import torch


def compute_perplexity(
    token_ids: torch.tensor, logits: torch.tensor, start: int, end: int
) -> list[float]:

    """Compute perplexity for predicted logits and groundtruth tokens.
    Args:
          token_ids: Token ids of input prompts (groundtruth)
          logits: Output logits from an LLM
          start: Index of the last input token to prefill
          end: Index of the last token that was processed by decode
    Returns:
          Dictionary of list of perplexities per prompt and
    """

    attention_mask = (
        (token_ids != 0).to(torch.int32).detach().clone().to(token_ids.device)
    )

    logits = logits[..., start + 1 : end + 1, :].contiguous()
    token_ids = token_ids[..., start + 1 : end + 1].contiguous()
    attention_mask = attention_mask[..., start + 1 : end + 1].contiguous()

    assert (
        token_ids.shape == logits.shape[0:2]
    ), "Mismatching token_ids and logits shapes, verify bs, seq_len dims match"

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    ## perplexity = e ^ (sum(losses) / num_tokenized_tokens)
    crossentropy_loss = (
        loss_fct(logits.transpose(1, 2), token_ids) * attention_mask
    ).sum(1)

    perplexity_batch = torch.exp(crossentropy_loss / attention_mask.sum(1)).tolist()
    perplexity_batch = [round(ppl, 6) for ppl in perplexity_batch]
    return perplexity_batch


def get_token_ids():
    """
    Default token ids for toy models
    """
    return [
        [3, 4, 5, 56, 76, 23, 2, 65, 49, 3, 98],
        [4, 65, 49, 32, 98, 5, 2, 23, 13, 58, 9],
        [5, 2, 23, 13, 58, 9, 3, 4, 5, 56, 76],
        [3, 4, 5, 65, 49, 32, 98, 5, 2, 23, 13],
    ]


def get_prompts(num_prompts: Optional[int] = None) -> list[str]:
    """Fetches prompts from the wikitext test dataset.
    Args:
          num_prompts: Number of prompts to fetch from dataset, will return all prompts if None
    """

    test_prompts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]

    num_test_prompts = 300

    random.seed(0)
    test_prompts = random.sample(test_prompts, num_test_prompts)

    # Ignore prompts that are: less than 20 tokens or a title or an incomplete sentence
    test_prompts = [
        s.replace("\n", "").rstrip()
        for s in test_prompts
        if sum(word.isalpha() for word in s.split()) > 20
        and s.count("=") < 2
        and s.split()[-1] == "."
    ][0:num_prompts]

    if num_prompts:
        test_prompts = test_prompts[0:num_prompts]

    return test_prompts


def get_prompt_lengths(
    token_ids: list[list[int]],
):
    max_length = 0
    lengths: list[int] = []
    for row in token_ids:
        lengths.append(len(row))
        max_length = max(max_length, len(row))

    return lengths, max_length


def pad_tokens(
    token_ids: list[list[int]],
    pad_to_multiple_of: int,
    pad_token: int = 0,
    device: torch.device | None = None,
) -> tuple[list[list[int]] | torch.Tensor, list[int] | torch.Tensor]:
    lengths, max_length = get_prompt_lengths(token_ids)
    if pad_to_multiple_of > 1:
        max_length = int(
            pad_to_multiple_of * math.ceil(max_length / pad_to_multiple_of)
        )
    for row in token_ids:
        pad_count = max_length - len(row)
        row.extend(pad_count * [pad_token])

    if device is not None:
        token_ids = torch.tensor(token_ids, device=device)
        lengths = torch.tensor(lengths, device=device)

    return token_ids, lengths


def calc_time(start=None, end=None, time_diff=None):
    """
    Caculates time difference between timestamps and prints it in desired time format
    """
    assert (start is None and end is None) ^ (
        time_diff is None
    ), "Pass either start and end timestamps or the time_diff"

    # Convert nanoseconds to seconds
    nanoseconds = time_diff or end - start
    total_seconds = nanoseconds / 1e9

    # Calculate hours, minutes, and seconds
    hours, remainder_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder_seconds, 60)

    if hours >= 1:
        time_taken = f"{int(hours):02d} hrs :{int(minutes):02d} mins :{round(float(seconds), 2):.2f} secs"
    elif minutes >= 1:
        time_taken = f"{int(minutes):02d} mins :{round(float(seconds), 2):.2f} secs"
    elif seconds >= 1:
        time_taken = f"{round(float(seconds), 2):.2f} secs"
    else:
        time_taken = f" {round(seconds * 1000, 3)} ms"
    return time_taken
