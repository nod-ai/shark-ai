# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


def max_negative_value(
    dtype: torch.dtype, device: torch.device | None = None
) -> torch.Tensor:
    """Returns a maximally negative value for the given dtype."""
    return torch.tensor(float("-inf"), dtype=dtype, device=device)


def create_attention_mask(
    boolean_input_mask: torch.Tensor,
    attention_dtype: torch.dtype,
    start_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generates a causal attention mask of [bs, 1, sl, sl] of activation dtype.

    All masked positions are -inf and unmasked are 0.0.

    The causal context mask will either be generated or use the initialization time buffer.
    Since this is a bool tensor of context_length^2, different deployment
    scenarios can benefit from managing this in different ways.
    """
    device = boolean_input_mask.device

    # Combine the causal context mask and input mask.
    dtype = (
        torch.float32 if attention_dtype == torch.float8_e4m3fnuz else attention_dtype
    )
    _, batch_seq_len = boolean_input_mask.shape

    causal_mask = create_causal_context_mask(
        src_len=batch_seq_len,
        target_len=batch_seq_len,
        start_positions=start_positions,
        device=device,
    )
    boolean_mask = torch.logical_or(causal_mask, boolean_input_mask[:, None, None, :])
    numeric_mask = torch.where(boolean_mask, max_negative_value(dtype, device), 0).to(
        dtype
    )
    return numeric_mask.to(device)


def create_attention_mask_for_decode(
    boolean_input_mask: torch.Tensor,
    attention_dtype: torch.dtype,
) -> torch.Tensor:
    device = boolean_input_mask.device

    dtype = (
        torch.float32 if attention_dtype == torch.float8_e4m3fnuz else attention_dtype
    )
    numeric_mask = torch.where(
        boolean_input_mask, max_negative_value(dtype, device), 0
    ).to(dtype)
    return numeric_mask.unsqueeze(1).unsqueeze(1).to(device)


def create_causal_context_mask(
    src_len: int,
    target_len: int,
    start_positions: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate a causal context mask of shape [1, 1, target_len, src_len].

    If start_positions is provided, it should be a tensor of shape [bs] indicating
    the starting position for each sequence in the batch. The mask will be adjusted
    accordingly to ensure that each position can only attend to previous positions
    in its own sequence.

    Args:
        src_len: Length of the source sequence.
        target_len: Length of the target sequence.
        start_positions: Optional tensor of shape [bs] indicating the starting position
                         for each sequence in the batch.
        device: The device to place the output mask on.
    """
    src = torch.arange(src_len, device=device)[None, None, None, :]
    target = torch.arange(target_len, device=device)[None, None, :, None]

    if start_positions is not None:
        target = target + start_positions[:, None, None, None]

    mask = src > target
    return mask


def create_boolean_chunked_attention_mask(
    attention_chunk_size: int,
    start_index: int,
    end_index: int,
    device: torch.device | None = None,
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
    return mask.to(device)


def create_chunked_attention_mask(
    attention_mask: torch.Tensor, attention_chunk_size: int
) -> torch.Tensor:
    """
    Apply a chunked attention mask onto a mask.

    This is a convenience function that combines the creation of the boolean
    chunked attention mask and its application to the provided attention mask.

    Args:
        attention_mask: The original attention mask of shape [bs, 1, sl, sl].
        attention_chunk_size: The size of each attention chunk.

    Returns:
        A new attention mask with chunked masking applied.
    """
    device = attention_mask.device
    batch_seq_len = attention_mask.shape[2]
    # TODO: handle decode step
    start_index = 0
    end_index = batch_seq_len
    chunked_boolean_attention_mask = create_boolean_chunked_attention_mask(
        attention_chunk_size=attention_chunk_size,
        # TODO: handle decode step
        start_index=start_index,
        end_index=end_index,
        device=device,
    )

    return torch.where(
        chunked_boolean_attention_mask,
        attention_mask,
        torch.tensor(
            max_negative_value(attention_mask.dtype, device=device),
            dtype=attention_mask.dtype,
        ),
    )


def create_input_mask(seq_lens: torch.Tensor, batch_seqlen: int) -> torch.Tensor:
    """
    Compute a boolean input mask for a batch of sequence lengths.

    The mask will be [bs, batch_seqlen] with True at any position that is masked.

    Args:
        seq_lens: [bs] tensor of integers representing the sequence lengths.
        batch_seqlen: The maximum sequence length in the batch.
    """
    range_vector = torch.arange(0, batch_seqlen, 1, device=seq_lens.device)
    matrix = seq_lens.unsqueeze(dim=-1)
    mask = range_vector >= matrix
    return mask
