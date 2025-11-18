# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Union

import re

from amdsharktank.types import *
from amdsharktank.utils.logging import transform_logger as logger

__all__ = [
    "MmtRHSShardingTransform",
]


class MmtRHSShardingTransform:
    """Shards tensors used as the RHS of a transposed matmul.

    Tensors matching any of the patterns will be split, if supported, into
    `num_shards`.
    """

    def __init__(
        self,
        *patterns: Union[str, re.Pattern],
        num_shards: int,
        skip_on_unsupported: bool = True,
    ):
        self.patterns = patterns
        self.num_shards = num_shards
        self.skip_on_unsupported = skip_on_unsupported

    def __call__(self, it: InferenceTensor):
        name = it.name
        if not any(re.match(p, name) for p in self.patterns):
            return it
        if isinstance(it, PrimitiveTensor):
            sharded = self._shard_primitive_tensor(it)
            if sharded is not None:
                return sharded

        if self.skip_on_unsupported:
            logger.debug("Skipping unsupported tensor: %r", it)
            return it
        else:
            raise ValueError(f"Unsupporting sharding for tensor: {it}")

    def _shard_primitive_tensor(
        self, pt: PrimitiveTensor
    ) -> Optional[list[PrimitiveTensor]]:
        t = pt.as_torch()
        shape = list(t.shape)
        if len(shape) < 2:
            return None
        shard_dim = 1
        shard_dim_size = shape[shard_dim]
        if (shard_dim_size % self.num_shards) != 0:
            return None
        shard_split_size = shard_dim_size // self.num_shards
        shard_ts = t.split(shard_split_size, dim=shard_dim)
        st = SplitPrimitiveTensor(
            name=pt.name, shape=pt.shape, shard_dim=shard_dim, ts=shard_ts
        )
        logger.debug("Sharding tensor %r -> %r", pt, st)
        return st

    def __repr__(self):
        return (
            f"ShardingTransform(num_shards={self.num_shards}, patterns={self.patterns})"
        )
