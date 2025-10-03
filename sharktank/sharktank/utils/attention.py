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
