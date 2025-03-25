# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
import time
import random
import re
from datetime import timedelta
from datasets import load_dataset


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
        if s != "" and len(s.split()) >= 20 and s.count("=") < 2
    ]

    if num_prompts:
        test_prompts = test_prompts[0:num_prompts]

    return test_prompts


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total_seconds = end - start
        time_taken = abs(timedelta(seconds=total_seconds))
        hours, minutes, seconds = re.split(":", str(time_taken))

        if total_seconds < 1:
            time_taken = f" {round(total_seconds * 1000, 3)} ms"
        elif total_seconds < 60:
            time_taken = "{:.2f} secs".format(round(float(total_seconds), 2))
        else:
            time_taken = "{:02d} hrs : {:02d} mins : {:.2f} secs".format(
                int(hours), int(minutes), round(float(seconds), 2)
            )

        return result

    return wrapper
