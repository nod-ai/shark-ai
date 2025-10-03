import argparse
import json
import tokenizers
from pathlib import Path
import numpy as np
import math

import torch
from sharktank.models.llm.config import ServiceConfig, KVCacheConfig
from sharktank.utils.llm_utils import (
    IreeInstance,
    LlmInstance,
    server_config_page_size,
    dtype_string_to_type,
)


def get_pages(bs: int, count: int):
    pages = torch.arange(start=1, stop=bs * count + 1, dtype=torch.int64)
    pages = pages.reshape(count, bs).T
    return pages


irpa_fp = "/shark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa"

export_dir = Path(".")
mlir_fp = export_dir / "model.mlir"
config_fp = export_dir / "model.json"
vmfb_fp = export_dir / "model.vmfb"

torch.random.manual_seed(12345)

with open(config_fp, "rt") as f:
    server_config = ServiceConfig(**json.loads(f.read()))
    server_config.paged_kv_cache = KVCacheConfig(**server_config.paged_kv_cache)

# Extract the running configuration:
page_kv_cache = server_config.paged_kv_cache
block_seq_stride = page_kv_cache.block_seq_stride
block_count = page_kv_cache.device_block_count
page_size = server_config_page_size(server_config)[0]

print("page_kv_cache", page_kv_cache)
kv_cache_dtype = "float16"
int_dtype = torch.int64

# Instantiate a single iree instance
iree_instance = IreeInstance(devices=["hip://0"], vmfb=vmfb_fp, parameters=irpa_fp)

cache_state = iree_instance.allocate(
    block_count, page_size, dtype=dtype_string_to_type[kv_cache_dtype], device_index=0
)

prefill_bs = iree_instance._prefill_bs

# Full prefill

# tokens = torch.tensor([
#       [128000, 128000, 8144, 264, 11944, 11, 14733, 44658, 3327, 1780, 315, 1268, 264, 6617, 3566, 2778, 4817, 24877, 4835, 11, 25998, 11, 21467, 11, 323, 17482, 6959, 13, 18230, 25, 74194, 38952, 11, 29807, 3996, 323, 274, 1224, 2690, 11, 43553, 2065, 11, 5665, 49100, 6373, 11, 23329, 323, 3221, 1773, 14399, 18468, 11, 2262, 23115, 11, 47801, 1963, 8246, 11, 23862, 17738, 320],
#       [128000, 128000, 8144, 264, 11944, 11, 14733, 44658, 3327, 1780, 315, 1268, 264, 6617, 3566, 2778, 4817, 24877, 4835, 11, 25998, 11, 21467, 11, 323, 17482, 6959, 13, 18230, 25, 74194, 38952, 11, 29807, 3996, 323, 274, 1224, 2690, 11, 43553, 2065, 11, 5665, 49100, 6373, 11, 23329, 323, 3221, 1773, 14399, 18468, 11, 2262, 23115, 11, 47801, 1963, 8246, 11, 23862, 17738, 320],
#       [0]*64,
#       [0]*64
# ], dtype=dtype)

tokens = torch.tensor(
    [
        128000,
        128000,
        8144,
        264,
        11944,
        11,
        14733,
        44658,
        3327,
        1780,
        315,
        1268,
        264,
        6617,
        3566,
        2778,
        4817,
        24877,
        4835,
        11,
        25998,
        11,
        21467,
        11,
        323,
        17482,
        6959,
        13,
        18230,
        25,
        74194,
        38952,
        11,
        29807,
        3996,
        323,
        274,
        1224,
        2690,
        11,
        43553,
        2065,
        11,
        5665,
        49100,
        6373,
        11,
        23329,
        323,
        3221,
        1773,
        14399,
        18468,
        11,
        2262,
        23115,
        11,
        47801,
        1963,
        8246,
        11,
        23862,
        17738,
        320,
    ],
    dtype=int_dtype,
)
tokens = tokens.repeat(prefill_bs, 1)

seq_lens = torch.tensor([len(tok) for tok in tokens], dtype=int_dtype)
max_len = max(seq_lens)
blocks = math.ceil(max_len / block_seq_stride)
blocked_len = blocks * block_seq_stride

assert tokens.shape[1] == blocked_len

pages = np.arange(start=1, stop=prefill_bs * blocks + 1, dtype=np.int64)
seq_block_ids = torch.tensor(pages.reshape(blocks, prefill_bs).T, dtype=int_dtype)

start_positions = torch.tensor([0] * prefill_bs, dtype=int_dtype)

print("*" * 50, "Full_prefill", "*" * 50)

print("tokens:   ", tokens.shape)
print("seq_block_ids:   ", seq_block_ids.shape, seq_block_ids)
print("cache_state:   ", cache_state.shape)
print("start_positions:   ", start_positions)
results = iree_instance.prefill(
    tokens, start_positions, seq_lens, seq_block_ids, cache_state
)
results = torch.asarray(np.asarray(results))

# if isinstance(results, tuple):
#     logits, indices = results
# else:
#     k = 8
#     logits = torch.asarray(np.asarray(results))
#     logits, indices = torch.topk(logits, k)

ctx_len = tokens.shape[1]

logits_full = results[:, :ctx_len]
# logits_full = logits[:, :ctx_len]
# indices_full = indices[:, :ctx_len]

print("logits_full:   ", logits_full.shape)
# print("indices_full:   ", indices_full.shape)

# Chunked prefill
print("*" * 50, "Chunked_prefill", "*" * 50)

iree_instance = IreeInstance(devices=["hip://0"], vmfb=vmfb_fp, parameters=irpa_fp)

cache_state = iree_instance.allocate(
    block_count, page_size, dtype=dtype_string_to_type[kv_cache_dtype], device_index=0
)

print("cache_state:   ", cache_state.shape)

prefill_bs = iree_instance._prefill_bs

start = 0
start_pos = start
num_reqs = 2
chunk_size = int(int(max_len) / num_reqs)

valid_bs = 2

for req_n in range(0, num_reqs):
    print("*" * 50, "prefill chunk", req_n, "*" * 50)
    end = start + chunk_size
    print("start, end, chunk_size:   ", start, end, chunk_size)
    tokens_chunk = tokens[:, start:end]
    max_len_chunk = tokens[:, :end].shape[1]
    blocks_chunk = math.ceil(max_len_chunk / block_seq_stride)
    blocked_len_chunk = blocks_chunk * block_seq_stride
    print("blocks_chunk", blocks_chunk)

    seq_block_ids_chunk = seq_block_ids[:, :blocks_chunk]
    seq_len_chunk = tokens_chunk.shape[1]
    start_positions_chunk = torch.tensor([start_pos] * prefill_bs)

    seq_lens_chunk = torch.full(
        [prefill_bs], (req_n + 1) * chunk_size, dtype=torch.int64
    )
    seq_lens_chunk = torch.minimum(seq_lens, seq_lens_chunk)
    logits_chunk = iree_instance.prefill(
        tokens_chunk,
        start_positions_chunk,
        seq_lens_chunk,
        seq_block_ids_chunk,
        cache_state,
    )
    logits_chunk = torch.asarray(np.asarray(logits_chunk))
    print(f"tokens_chunk.shape = {tokens_chunk.shape}")
    print(f"start_positions_chunk = {start_positions_chunk}")
    print(f"seq_lens_chunk = {seq_lens_chunk}")
    print(f"seq_block_ids_chunk.shape = {seq_block_ids_chunk.shape}")
    print(f"cache_state.shape = {cache_state.shape}")
    print(f"logits_chunk.shape = {logits_chunk.shape}")

    # if isinstance(results, tuple):
    #     logits, indices = results
    # else:
    #     k = 8
    #     logits = torch.asarray(np.asarray(results))
    #     logits, indices = torch.topk(logits, k)

    # ctx_len = seq_len_chunk

    # logits_chunk = logits[:, :ctx_len]
    # indices_chunk = indices[:, :ctx_len]

    # print("logits_chunk:   ", logits_chunk.shape)
    # print("indices_chunk:   ", indices_chunk.shape)
    # print(
    #     "start_pos, end_pos",
    #     start,
    #     end,
    # )

    logits_excepted = logits_full[:, start:end]

    for bs in range(prefill_bs):
        try:
            torch.testing.assert_close(
                logits_excepted[bs, :, :],
                logits_chunk[bs, :, :],
                atol=0,
                rtol=0,
            )

        except AssertionError as error:
            print(f"\n\nFAILED *Logits*: bs={bs} / chunk={req_n}: \n{error}")
        else:
            print(f"\nPASSED *Logits*: bs={bs} / chunk={req_n}")

        # try:
        #     torch.testing.assert_close(
        #         indices_full[bs, start_pos : start_pos + ctx_len],
        #         indices_chunk[bs, :, :],
        #         atol=0,
        #         rtol=0,
        #     )

        # except AssertionError as error:
        #     print(f"\n\nFAILED *Indices*: bs={bs} / chunk={req_n}: \n{error}")
        # else:
        #     print(f"\nPASSED *Indices*: bs={bs} / chunk={req_n}")

    start = end
    start_pos = start
