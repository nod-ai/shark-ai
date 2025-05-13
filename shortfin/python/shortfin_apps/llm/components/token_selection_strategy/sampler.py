# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass

import numpy as np
import random

from dataclasses import dataclass
from typing import Dict, List

import shortfin as sf
import shortfin.array as sfnp

from shortfin_apps.utils import convert_int_to_float

from .config import SamplingKernels


@dataclass
class GPUSampler:

    sampling_kernels: Dict[str, sf.ProgramFunction]

    async def select_greedy(self, logits: sfnp.device_array, fiber: sf.Fiber) -> int:
        """Greedily select a single token using `argmax`.

        Args:
            logits (sfnp.device_array): Logits from decode.
            beam_id (int): Beam ID.

        Returns:
            int: Max token.
        """
        argmax_fn = self.sampling_kernels[SamplingKernels.ARGMAX.value]

        (argmax_result,) = await argmax_fn(logits, fiber=fiber)
        argmax_result_host = argmax_result.for_transfer()
        argmax_result_host.copy_from(argmax_result)

        await logits.device
        return argmax_result_host.items.tolist()[0]


@dataclass
class CPUSampler:
    def sample_top_k(self, tokens: List[int], probs: List[float], k: int):
        choices: List[int] = random.choices(tokens, weights=probs, k=k)
        token_prob_map = dict(zip(tokens, probs))
        return choices, [token_prob_map[token] for token in choices]

    def sample_top_p(self, tokens: List[int], probs: List[float], p: float, k: int):
        token_prob_map = dict(zip(tokens, probs))
        sorted_probs = sorted(token_prob_map.items(), key=lambda x: x[1], reverse=True)

        cumulative_prob = 0.0
        selected_tokens = []
        selected_probs = []
        for token, prob in sorted_probs:
            cumulative_prob += prob
            selected_tokens.append(token)
            selected_probs.append(prob)
            if cumulative_prob > p:
                break

        choices: List[int] = random.choices(tokens, weights=probs, k=k)
        return choices, [token_prob_map[token] for token in choices]

    def select_top_k(self, logits: sfnp.device_array, k: int):
        """
        This function is used to get the top k tokens and their cumulative probabilities.
        """
        partitioned_tokens = np.argpartition(logits, k)
        # Slice off all axes except the last one
        zero_indices = (0,) * (partitioned_tokens.ndim - 1)

        # Obtain tokens & values from partition
        top_tokens = partitioned_tokens[zero_indices + (slice(k, None),)].tolist()
        values_view = logits.view(*zero_indices).items

        top_values = []
        for token in top_tokens:
            value = values_view[token]
            if isinstance(value, int):
                value = convert_int_to_float(value, logits.dtype)

            top_values.append(value)

        return top_tokens, top_values

    def select_greedy(self, logits: sfnp.device_array) -> int:
        """Greedily select a single token using `argmax`.

        Args:
            logits (sfnp.device_array): Logits from decode.

        Returns:
            int: Max token.
        """
        token = sfnp.argmax(logits)
        token_int = token.items[0]
        return token_int
