# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from iree.turbine.aot import *

from amdsharktank.types import SplitPrimitiveTensor
from amdsharktank.ops import reshard_split, replicate
from amdsharktank.layers.paged_attention import PagedGQAttention, build_cache
from amdsharktank.utils import cli


def main():
    parser = cli.create_parser()
    parser.add_argument(
        "--output-mlir",
        help="Output file path for exported MLIR file",
        default="/tmp/kv_cache.mlir",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        help="Batch size to generate, e.g. `4` or `2`",
        type=lambda arg: int(arg),
        default="2",
    )
    parser.add_argument(
        "--sharding",
        help="Sharding level of kv-cache",
        type=lambda arg: int(arg),
        default="1",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Include verbose logging",
        action="store_true",
    )
    parser.add_argument(
        "--strict",
        help="Enables strictness during export",
        action="store_true",
    )

    args = cli.parse(parser)

    bs = args.batch_size

    bs = 4
    seq_length = 24
    attn_head_count = 4
    attn_head_dim = 16
    transformer_block_count = 4
    block_seq_stride = 4
    page_count = bs * seq_length // block_seq_stride
    write_seq_length = seq_length - 4

    kv_cache = build_cache(
        transformer_block_count=transformer_block_count,
        attn_head_count=attn_head_count,
        attn_head_dim=attn_head_dim,
        block_seq_stride=block_seq_stride,
        cache_dtype=torch.float32,
    )
    cache = PagedGQAttention(
        kv_cache=kv_cache,
        transformer_block_index=1,
        attn_dtype=torch.float32,
        use_rope=False,
        attention_chunk_size=None,
    )

    alloc = cache.allocate(page_count=page_count)
    allocation = alloc

    model = torch.nn.Module()
    fxb = FxProgramsBuilder(model)

    page_ids = torch.empty(bs, seq_length // block_seq_stride, dtype=torch.int64)
    write_page_ids = page_ids[:, : write_seq_length // block_seq_stride]
    partition_0 = torch.empty(
        (bs, write_seq_length, attn_head_count, attn_head_dim), dtype=torch.float32
    )

    if args.sharding > 1:
        partition_0 = reshard_split(partition_0, dim=2, count=args.sharding).shards
        allocation = allocation[0].shards

    argument_device_affinities = {}
    for i in range(args.sharding):
        argument_device_affinities[i] = DeviceAffinity(f"{i}")
        argument_device_affinities[i + args.sharding] = DeviceAffinity(f"{i}")

    @fxb.export_program(
        name="write",
        args=(allocation, partition_0, write_page_ids),
        strict=False,
        argument_device_affinities=argument_device_affinities,
    )
    def _(model, state, partition_0, write_page_ids: torch.Tensor) -> torch.Tensor:
        old_state = state
        if args.sharding > 1:
            state = [SplitPrimitiveTensor(ts=state, shard_dim=alloc[0].shard_dim)]
            partition_0 = SplitPrimitiveTensor(ts=partition_0, shard_dim=2)
            write_page_ids = replicate(write_page_ids, count=args.sharding)
        cache.write(
            state,
            cache_partitions=[partition_0, partition_0],
            transformer_block_index=1,
            page_ids=write_page_ids,
        )
        return state

    if args.verbose:
        for name, ep in fxb.programs.items():
            print(f"EXPORT {name}:\n{ep}")

    print("Exporting")
    output = export(fxb)
    print(f"Saving to '{args.output_mlir}'")
    output.save_mlir(args.output_mlir)


if __name__ == "__main__":
    main()
