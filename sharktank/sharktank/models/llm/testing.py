import torch

from .llm import PagedLlmModelV1
from sharktank.utils.math import round_up_to_multiple_of
from typing import Any, Tuple, OrderedDict


def make_random_prefill_args(
    self, model: PagedLlmModelV1, batch_size: int
) -> OrderedDict[str, Any]:
    # TODO: use this in sharded llama tests as well.
    seq_lens = torch.tensor([14, 9, self.block_seq_stride - 1], dtype=torch.int64)
    batch_seq_len = round_up_to_multiple_of(
        int(torch.max(self.prefill_seq_lens)), model.cache.pad_sequence_stride
    )
    token_ids = torch.randint(
        low=0,
        high=self.vocabulary_size,
        size=[self.batch_size, batch_seq_len],
        dtype=torch.int32,
    )
    attention_mask = [
        model.attention_mask(model.input_mask(self.prefill_seq_lens, batch_seq_len))
    ]
    seq_block_ids = [
        torch.arange(
            self.batch_size * batch_seq_len // self.config.block_seq_stride
        ).view(self.batch_size, -1)
    ]
    cache_state = model.cache.allocate(page_count=self.cache_page_count)
    cache_state = [torch.rand_like(cache_state[0])]
    return OrderedDict(
        [
            ("tokens", token_ids),
            ("attention_mask", attention_mask),
            ("seq_block_ids", seq_block_ids),
            ("cache_state", cache_state),
        ]
    )
