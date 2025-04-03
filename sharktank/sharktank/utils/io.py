# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import torch
from os import PathLike
from contextlib import closing
import socket

from iree.turbine.aot import ParameterArchiveBuilder, ParameterArchive


class ShardedArchiveBuilder(ParameterArchiveBuilder):
    """A ParameterArchiveBuilder that can contain subordinate builders for each rank.

    TODO: This currently collects all data in memory and commits at once. This
    can be made much more memory efficient for computed datasets by exposing
    a Python streaming save helper upstream and using that.
    """

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path
        self._rank_builders: dict[int, ParameterArchiveBuilder] = {}

    def for_rank(self, rank: int) -> ParameterArchiveBuilder:
        """Returns a ParameterArchiveBuilder for tensors specific to the given rank."""
        if rank in self._rank_builders:
            return self._rank_builders[rank]
        b = ParameterArchiveBuilder()
        self._rank_builders[rank] = b
        return b

    def commit(self):
        """Performs final commit of all builders to disk."""
        self.save(self.save_path)
        for i, rank_builder in self._rank_builders.items():
            rank_builder.save(ShardedArchiveBuilder.path_for_rank(self.save_path, i))

    @staticmethod
    def path_for_rank(path: Path, rank: int):
        """Returns a path to a file modified with a rank.

        Example input:
          /tmp/foobar.irpa

        Example output:
          /tmp/foobar.rank0.irpa
        """
        return path.with_suffix(f".rank{rank}{path.suffix}")


def save_tensor_as_irpa(tensor: torch.Tensor, path: PathLike):
    """Save a single tensor into an IRPA file."""
    param_builder = ParameterArchiveBuilder()
    param_builder.add_tensor("", tensor)
    param_builder.save(path)


def load_irpa_as_tensor(tensor: torch.Tensor, path: PathLike, **kwargs):
    """Load a tensor form an IRPA file that holds only one tensor."""
    params = ParameterArchive(path, **kwargs)
    items = params.items()
    if len(items) != 1:
        raise ValueError(
            f'Too many items {len(items)} in IRPA file "{path}".'
            " Only a single tensor was expected."
        )
    return items[0][1].as_tensor()


def find_free_port() -> int:
    """This tries to find a free port.

    Race conditions are possible - the port can be acquired between when this
    runs and when it is used.

    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
