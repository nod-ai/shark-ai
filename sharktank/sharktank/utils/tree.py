# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Any, Callable, Mapping, Iterable, Sequence, Union
from collections.abc import Mapping as MappingABC, Iterable as IterableABC

Key = Any
Leaf = Any
Tree = Mapping[Key, Union[Leaf, "Tree"]] | Iterable[Union[Leaf, "Tree"]] | Leaf
IsLeaf = Callable[[Tree], bool]


def is_leaf_default(tree: Tree) -> bool:
    return not isinstance(tree, IterableABC)


def is_not_tuple_list_or_dict(tree: Tree) -> bool:
    return not isinstance(tree, (tuple, list, dict))


def map_nodes(
    tree: Tree, f: Callable[[Tree], Tree], is_leaf: IsLeaf | None = None
) -> Tree:
    """Apply `f` for each node in the tree. Leaves and branches.

    This includes the root `tree` as well."""
    if is_leaf is None:
        is_leaf = is_leaf_default

    if is_leaf(tree):
        return f(tree)
    elif isinstance(tree, MappingABC):
        return f({k: map_nodes(v, f, is_leaf) for k, v in tree.items()})
    else:
        return f([map_nodes(v, f, is_leaf) for v in tree])


def map_leaves(
    tree: Tree, f: Callable[[Tree], Tree], is_leaf: IsLeaf | None = None
) -> Tree:
    """Apply `f` for each leaf in the tree."""
    if is_leaf is None:
        is_leaf = is_leaf_default

    def f_leaves_and_nodes(t: Tree) -> Tree:
        if is_leaf(t):
            return f(t)
        else:
            return t

    return map_nodes(tree, f_leaves_and_nodes, is_leaf)


def flatten(tree: Tree, is_leaf: IsLeaf | None = None) -> Sequence[Leaf]:
    """Get the leaves of the tree."""
    return [x for x in _flatten(tree, is_leaf)]


def _flatten(tree: Tree, is_leaf: IsLeaf | None = None) -> Iterable[Leaf]:
    if is_leaf is None:
        is_leaf = is_leaf_default
    if is_leaf(tree):
        yield tree
    elif isinstance(tree, MappingABC):
        for v in tree.values():
            yield from flatten(v, is_leaf)
    else:
        for v in tree:
            yield from flatten(v, is_leaf)
