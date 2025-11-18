# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from amdsharktank.types import (
    ReplicatedTensor,
    Theta,
)
from amdsharktank.utils.attention import *
from .base import (
    ThetaLayer,
)


class BaseCausalLMModel(ThetaLayer):
    """Base class for causal LM models.

    This provides some utilities and common API surface related to masking
    and token extraction.

    It presumes a fixed context length.
    """

    def __init__(
        self,
        theta: Theta,
        *,
        context_length: int,
        device: Optional[torch.device] = None,
        activation_dtype: torch.dtype = torch.float32,
        attention_dtype: torch.dtype = torch.float32,
        fake_quant: bool = True,
    ):
        super().__init__(theta)
        self.device = device
        self.activation_dtype = activation_dtype
        self.attention_dtype = attention_dtype
        self.context_length = context_length
        self.fake_quant = fake_quant
        self.causal_context_mask = None

    def extract_tokens_from_logits(
        self, logits: torch.Tensor, seq_lens: list[int]
    ) -> list[int]:
        """Extracts tokens from a batch of logits (B, S, D).

        The length of seq_lens must be equal to the batch size.
        Note that there are ways to code the indexing as tensor operations
        but it just creates a bunch of weirdly shaped little work on the
        accelerator. Statically looping like this is more efficient.
        """
        bs, *_ = logits.shape
        assert len(seq_lens) == bs
        results = []
        for batch, seq_len in enumerate(seq_lens):
            step_logits = logits[batch, seq_len - 1]
            results.append(torch.argmax(step_logits))
        return results
