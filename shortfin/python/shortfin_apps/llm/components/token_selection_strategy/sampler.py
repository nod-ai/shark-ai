# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from concurrent.futures import ThreadPoolExecutor, Future, wait
import logging
import numpy as np

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union

import shortfin.array as sfnp


logger = logging.getLogger(__name__)


class SamplerThreadPool:
    executor: None | ThreadPoolExecutor = None

    def __init__(self, max_workers: int = 128):
        """Initialize the thread pool with a specified number of workers.

        Args:
            max_workers (int, optional): Maximum number of worker threads. Defaults to 128.
        """
        if SamplerThreadPool.executor is None:
            SamplerThreadPool.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.max_workers = max_workers

    def submit(self, func: Callable, *args) -> Future:
        """Submit a task to the thread pool.

        Args:
            func (Callable): Function to run.
            *args: Positional arguments for the function.

        Returns:
            Future: Future for when task is done.
        """
        return SamplerThreadPool.executor.submit(func, args)

    def submit_and_wait(self, func: Callable, *args) -> Future:
        """Submit a task to the thread pool and wait for it to complete.

        Args:
            func (Callable): Function to run.
            *args: Positional arguments for the function.

        Returns:
            Future: Future for when task is done.
        """
        future = SamplerThreadPool.executor.submit(func, args)
        return future.result()

    def submit_batch(self, funcs: List[Tuple[Callable, Any]]) -> Future:
        """Submit a batch of tasks to the thread pool.

        Args:
            funcs (List[Tuple[Callable, Any]]): An iterable of tuples, where each tuple contains a function and its arguments.

        Returns:
            Future: A Future object representing the execution of the batch of tasks.
        """
        return [SamplerThreadPool.executor.submit(func, args) for func, args in funcs]

    def submit_batch_and_wait(self, funcs: List[Tuple[Callable, Any]]) -> Future:
        """Submit a batch of tasks to the thread pool and wait for all to complete.

        Args:
            funcs (List[Tuple[Callable, Any]]): An iterable of tuples, where each tuple contains a function and its arguments.

        Returns:
            Future: A Future object representing the execution of the batch of tasks.
        """
        return wait(self.submit_batch(funcs))


@dataclass
class Sampler:
    def sample_top_k(self, tokens: np.array, probs: np.array, k: int):
        p = probs / probs.sum()

        choices = np.random.choice(tokens, size=k, replace=True, p=p)

        token_to_p = {int(t): float(p_) for t, p_ in zip(tokens, p)}
        chosen_probs = np.array([token_to_p[int(t)] for t in choices])

        return choices, chosen_probs

    def sample_top_p(
        self,
        tokens: np.array,
        probs: np.array,
        p: float,
        k: int,
        return_probs=False,
    ):
        cum = np.cumsum(probs)
        idx = np.searchsorted(cum, p, side="right") + 1

        tokens, probs = tokens[:idx], probs[:idx]

        weights = probs / probs.sum()

        choices = np.random.choice(tokens, size=k, p=weights)
        chosen_probs = None
        if return_probs:
            prob_map = {tok: pr for tok, pr in zip(tokens, probs)}
            chosen_probs = np.array([prob_map[t] for t in choices])

        return choices, chosen_probs

    def select_top_k(
        self,
        logits: Union[np.array, sfnp.device_array],
        indices: Union[np.array, sfnp.device_array, None],
        k: int,
    ) -> Tuple[np.array, np.array]:
        """
        This function is used to get the top k tokens and their cumulative probabilities.
        """
        if isinstance(logits, sfnp.device_array):
            logits = np.array(logits)

        if isinstance(indices, sfnp.device_array):
            indices = np.array(indices)

        # Slice off all axes except the last one
        zero_indices = (0,) * (logits.ndim - 1)

        if indices is not None:
            tokens_index = zero_indices + (slice(None, -k),)
            return indices[tokens_index], logits[tokens_index]

        tokens_index = zero_indices + (slice(k, None),)
        partitioned_tokens = np.argpartition(logits, k)

        # Obtain tokens & values from partition
        top_tokens = partitioned_tokens[tokens_index]
        top_values = np.take(logits, top_tokens, axis=-1)[zero_indices]

        return top_tokens, top_values

    def select_greedy(self, logits: sfnp.device_array) -> int:
        """Greedily select a single token using `argmax`.

        Args:
            logits (sfnp.device_array): Logits from decode.

        Returns:
            int: Max token.
        """
        token = np.argmax(logits).item()
        return token
