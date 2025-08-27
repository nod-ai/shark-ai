# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from sharktank.types import (
    ReplicatedTensor,
    Theta,
)

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

    def _maximally_negative_value(self, dtype) -> torch.Tensor:
        """Returns a maximally negative value for the given dtype.

        This can be overriden to decide on a different policy.
        """
        return torch.tensor(float("-inf"), dtype=dtype, device=self.device)

    def generate_causal_context_mask(
        self, src_len, target_len, start_positions
    ) -> torch.Tensor:
        src = torch.arange(src_len)[None, None, None, :]
        target = torch.arange(target_len)[None, None, :, None]

        if start_positions is not None:
            target = target + start_positions[:, None, None, None]

        causal_context_mask = src > target

        return causal_context_mask.to(self.device)

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
        start_positions: torch.Tensor | None = None,
    ):
        """Generates a causal attention mask of [bs, 1, sl, sl] of activation dtype.

        All masked positions are -inf and unmasked are 0.0.

        The causal context mask will either be generated or use the initialization time buffer.
        Since this is a bool tensor of context_length^2, different deployment
        scenarios can benefit from managing this in different ways.
        """

        # Combine the causal context mask and input mask.
        dtype = (
            torch.float32
            if self.attention_dtype == torch.float8_e4m3fnuz
            else self.attention_dtype
        )
        _, batch_seq_len = input_mask.shape

        causal_mask = self.generate_causal_context_mask(
            src_len=batch_seq_len,
            target_len=batch_seq_len,
            start_positions=start_positions,
        )
        boolean_mask = torch.logical_or(causal_mask, input_mask[:, None, None, :])
        numeric_mask = torch.where(
            boolean_mask, self._maximally_negative_value(dtype), 0
        ).to(dtype)
        return numeric_mask.to(self.device)

    def chunked_attention_mask(
        self, attention_mask: torch.Tensor | ReplicatedTensor
    ) -> torch.Tensor:
        """Apply a chunked attention mask onto a mask."""
        batch_seq_len = attention_mask.shape[2]
        # TODO: handle decode step
        start_index = 0
        end_index = batch_seq_len
        chunked_boolean_attention_mask = self.create_boolean_chunked_attention_mask(
            attention_chunk_size=self.config.attention_chunk_size,
            # TODO: handle decode step
            start_index=start_index,
            end_index=end_index,
        ).to(attention_mask.device)

        return torch.where(
            chunked_boolean_attention_mask,
            attention_mask,
            torch.tensor(
                self._maximally_negative_value(attention_mask.dtype),
                dtype=attention_mask.dtype,
            ),
        )

    def create_boolean_chunked_attention_mask(
        self, attention_chunk_size: int, start_index: int, end_index: int
    ) -> torch.Tensor:
        """
        Generate the following:

        'What'      :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚    |
        '▁is'       :  1 ■ ■ ⬚ ⬚ ⬚ ⬚     |
        '▁ch'       :  2 ■ ■ ■ ⬚ ⬚ ⬚     |
        'unked'     :  3 ⬚ ⬚ ⬚ ■ ⬚ ⬚    |
        '▁attention':  4 ⬚ ⬚ ⬚ ■ ■ ⬚    |
        '?'         :  5 ⬚ ⬚ ⬚ ■ ■ ■     |

        If the chunk size is 3.
        This can just be applied over the already created attention mask

        ⬚ - masked (False).
        ■ - unmasked (True).
        """
        arange_vector = torch.arange(start_index, end_index)
        block_pos = torch.abs(
            arange_vector.unsqueeze(0) // attention_chunk_size
            - arange_vector.unsqueeze(1) // attention_chunk_size
        )
        token_pos = arange_vector.unsqueeze(0) - arange_vector.unsqueeze(1)
        mask = (block_pos == 0) & (token_pos <= 0)
        return mask

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
