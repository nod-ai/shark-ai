import shortfin.array as sfnp

PrefillHostCacheType = tuple[
    # Cache for tokens_host
    dict[int, sfnp.device_array],
    # seq_lens_host
    dict[int, sfnp.device_array],
    # seq_block_ids_host
    dict[tuple[int, int], sfnp.device_array],
    # logits_host
    dict[tuple[int, int], sfnp.device_array],
    # indices_host
    dict[tuple[int, int], sfnp.device_array],
]

DecodeHostCacheType = tuple[
    # Cache for tokens_host
    dict[int, sfnp.device_array],
    # seq_lens_host
    dict[int, sfnp.device_array],
    # start_positions_host
    dict[int, sfnp.device_array],
    # seq_block_ids_host
    dict[tuple[int, int], sfnp.device_array],
    # logits_host
    dict[tuple[int, int], sfnp.device_array],
    # indices_host
    dict[tuple[int, int], sfnp.device_array],
]
