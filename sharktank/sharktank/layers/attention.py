# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from abc import ABC, abstractmethod
from sharktank.types import AnyTensor


__all__ = ["CacheAllocation", "KVCache"]


class CacheAllocation:
    def __init__(
        self, allocation: list[torch.Tensor], devices: list[int] | None = None
    ):
        devices = devices if devices is not None else list(range(len(allocation)))
        assert len(devices) == len(allocation)

        self.allocation = allocation

        from iree.turbine.aot import DeviceAffinity

        self.device_affinities = [DeviceAffinity(device) for device in devices]

    def __len__(self):
        return len(self.allocation)

    def __getitem__(self, idx):
        return self.allocation[idx]


class KVCache(ABC):
    @abstractmethod
    def allocate(self, page_count: int) -> CacheAllocation:
        """Allocates the cache state for a given number of pages."""
        ...

    @property
    @abstractmethod
    def state_count(self) -> int:
        ...

    @abstractmethod
    def unflatten_page_table(self, state: CacheAllocation) -> list[torch.Tensor]:
        ...

    @abstractmethod
    def read(
        self,
        state: CacheAllocation,
        *,
        transformer_block_index: int,
        page_ids: AnyTensor,
    ) -> AnyTensor:
        ...

    @abstractmethod
    def write(
        self,
        state: CacheAllocation,
        *,
        cache_partitions: list[AnyTensor],
        transformer_block_index: int,
        page_ids: AnyTensor,
        start_positions: AnyTensor | None,
    ) -> None:
        ...

    @abstractmethod
    def write_timestep(
        self,
        state: CacheAllocation,
        *,
        cache_partitions: list[AnyTensor],
        transformer_block_index: int,
        seq_positions: AnyTensor,
        page_ids: AnyTensor,
    ) -> None:
        ...
