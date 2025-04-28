# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from sharktank.types import Theta
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
        static_tables: bool = True,
        static_context_mask: bool = False,
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

        if static_tables:
            self.register_buffer(
                "causal_context_mask", self.generate_causal_context_mask()
            )
        else:
            self.causal_context_mask = None

    def _assert_device(self, *ts: torch.Tensor, dtype: Optional[torch.dtype] = None):
        if self.device is not None:
            for t in ts:
                assert (
                    t.device == self.device
                ), f"Expected tensor to be on device {self.device} but it is on {t.device}"
                if dtype is not None:
                    assert (
                        t.dtype == dtype
                    ), f"Expected tensor to have dtype {dtype} but it is {t.dtype}"

    def _maximally_negative_value(self, dtype):
        """Returns a maximally negative value for the given dtype.

        This can be overriden to decide on a different policy.
        """
        return float("-inf")

    def generate_causal_context_mask(self) -> torch.Tensor:
        context_length = self.context_length
        unary_broadcast_ones = torch.ones([1, 1], dtype=torch.bool, device=self.device)
        context_broadcast_ones = unary_broadcast_ones.expand(
            context_length, context_length
        )
        causal_context_mask = torch.triu(
            context_broadcast_ones,
            diagonal=1,
        )[None, None, :, :]
        return causal_context_mask

    def input_mask(
        self,
        # [bs] of integers
        seq_lens: torch.Tensor,
        batch_seqlen: int,
    ):
        """Compute a boolean input mask for a batch of sequence lengths.

        The mask will be [bs, batch_seqlen] with True at any position that is
        masked.
        """
        range_vector = torch.arange(0, batch_seqlen, 1, device=self.device)
        matrix = seq_lens.unsqueeze(dim=-1)
        mask = range_vector >= matrix
        return mask

    def decode_attention_mask(self, boolean_input_mask: torch.Tensor):
        dtype = (
            torch.float32
            if self.attention_dtype == torch.float8_e4m3fnuz
            else self.attention_dtype
        )
        numeric_mask = torch.where(
            boolean_input_mask, self._maximally_negative_value(dtype), 0
        ).to(dtype)
        return numeric_mask.unsqueeze(1).unsqueeze(1).to(self.device)

    def attention_mask(
        self,
        input_mask: torch.Tensor,
        *,
        causal_context_mask: Optional[torch.Tensor] = None,
    ):
        """Generates a causal attention mask of [1, 1, sl, sl] of activation dtype.

        All masked positions are -inf and unmasked are 0.0.

        The pre-initialized causal context mask can be passed in. If not, then
        it will either be generated or use the initialization time buffer.
        Since this is a bool tensor of context_length^2, different deployment
        scenarios can benefit from managing this in different ways.
        """
        if causal_context_mask is None:
            # Try to use the statically generated.
            causal_context_mask = self.causal_context_mask
        if causal_context_mask is None:
            # Fallback to dynamically generated.
            causal_context_mask = self.generate_causal_context_mask()

        # Combine the causal context mask and input mask.
        dtype = (
            torch.float32
            if self.attention_dtype == torch.float8_e4m3fnuz
            else self.attention_dtype
        )
        _, batch_seq_len = input_mask.shape
        causal_mask = causal_context_mask[:, :, :batch_seq_len, :batch_seq_len]
        boolean_mask = torch.logical_or(causal_mask, input_mask[:, None, None, :])
        numeric_mask = torch.where(
            boolean_mask, self._maximally_negative_value(dtype), 0
        ).to(dtype)
        return numeric_mask.to(self.device)

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
            if step_logits.dtype == torch.float8_e4m3fnuz:
                step_logits_as_int8 = step_logits.view(dtype=torch.int8)
                results.append(torch.argmax(step_logits_as_int8))
            else:
                results.append(torch.argmax(step_logits))
        return results
