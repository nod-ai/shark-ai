import torch

from .llm import PagedLlmModelV1
from sharktank.utils.math import round_up_to_multiple_of
from sharktank.utils.attention import *
from typing import Any, Tuple, OrderedDict


def make_random_decode_args(
    model: PagedLlmModelV1, batch_size: int
) -> OrderedDict[str, Any]:
    prefill_seq_lens = torch.randint(
        size=[batch_size],
        low=1,
        high=min(
            2 * model.config.block_seq_stride,
            model.config.hp.context_length,
        )
        - 1,
        dtype=torch.int64,
        device=model.device,
    )

    start_positions = [prefill_seq_lens]
    seq_lens = prefill_seq_lens + 1
    batch_seq_len = round_up_to_multiple_of(
        int(torch.max(seq_lens)), model.config.block_seq_stride
    )
    decode_token_ids = torch.randint(
        low=0,
        high=model.config.hp.vocab_size,
        size=[batch_size, 1],
        dtype=torch.int32,
    )
    seq_block_ids = [
        torch.arange(batch_size * batch_seq_len // model.config.block_seq_stride).view(
            batch_size, -1
        )
    ]
    cache_state = model.cache.allocate(page_count=seq_block_ids[0].numel() + batch_size)
    cache_state = [torch.rand_like(cache_state[0])]
    return OrderedDict(
        [
            ("tokens", decode_token_ids),
            ("seq_lens", seq_lens),
            ("start_positions", start_positions),
            ("seq_block_ids", seq_block_ids),
            ("cache_state", cache_state),
        ]
    )


def make_random_prefill_args(
    model: PagedLlmModelV1, batch_size: int
) -> OrderedDict[str, Any]:
    seq_lens = torch.randint(
        size=[batch_size],
        low=1,
        high=min(
            2 * model.config.block_seq_stride,
            model.config.hp.context_length,
        )
        - 1,
        dtype=torch.int64,
        device=model.device,
    )
    batch_seq_len = round_up_to_multiple_of(
        int(torch.max(seq_lens)), model.config.block_seq_stride
    )
    token_ids = torch.randint(
        low=0,
        high=model.config.hp.vocab_size,
        size=[batch_size, batch_seq_len],
        dtype=torch.int32,
        device=model.device,
    )

    seq_block_ids = [
        torch.arange(
            batch_size * batch_seq_len // model.config.block_seq_stride,
            device=model.device,
        ).view(batch_size, -1)
    ]
    cache_state = model.cache.allocate(page_count=seq_block_ids[0].numel() + batch_size)
    cache_state = [torch.rand_like(cache_state[0])]
    return OrderedDict(
        [
            ("tokens", token_ids),
            ("seq_lens", seq_lens),
            ("seq_block_ids", seq_block_ids),
            ("cache_state", cache_state),
        ]
    )
