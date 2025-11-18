# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Any, Callable, Mapping, Iterable, Sequence, Union
from collections.abc import Mapping as MappingABC, Iterable as IterableABC
import functools

from amdsharktank.utils.misc import (
    assert_equal as default_assert_elements_equal,
    assert_sets_equal,
)

Key = Any
Path = tuple[Key, ...]
Leaf = Any
Tree = Mapping[Key, Union[Leaf, "Tree"]] | Iterable[Union[Leaf, "Tree"]] | Leaf
IsLeaf = Callable[[Tree], bool]


def is_leaf_default(tree: Tree) -> bool:
    return not isinstance(tree, IterableABC) or isinstance(tree, str)


def is_not_tuple_list_or_dict(tree: Tree) -> bool:
    return not isinstance(tree, (tuple, list, dict))


def assert_equal(
    tree1: Tree,
    tree2: Tree,
    /,
    is_leaf: IsLeaf = is_leaf_default,
    assert_equal: Callable[[Leaf, Leaf], None] = default_assert_elements_equal,
):
    dict1 = dict(flatten_with_paths(tree1, is_leaf))
    dict2 = dict(flatten_with_paths(tree2, is_leaf))

    try:
        assert_sets_equal(set(dict1.keys()), set(dict2.keys()))
    except AssertionError as ex:
        raise AssertionError(f"Tree structure not equal") from ex

    for k1, v1 in dict1.items():
        try:
            assert_equal(v1, dict2[k1])
        except AssertionError as ex:
            raise AssertionError(
                f"Trees not equal: elements in trees with path {k1} not equal"
            ) from ex


def map_nodes(
    tree: Tree,
    f: Callable[[Tree], Tree],
    is_leaf: IsLeaf = is_leaf_default,
    *,
    dict_type: type = dict,
    sequence_type: type = tuple,
) -> Tree:
    """Apply `f` for each node in the tree. Leaves and branches.

    This includes the root `tree` as well."""
    if is_leaf(tree):
        return f(tree)
    elif isinstance(tree, MappingABC):
        return f(
            dict_type(
                (
                    k,
                    map_nodes(
                        v, f, is_leaf, dict_type=dict_type, sequence_type=sequence_type
                    ),
                )
                for k, v in tree.items()
            )
        )
    else:
        return f(
            sequence_type(
                map_nodes(
                    v, f, is_leaf, dict_type=dict_type, sequence_type=sequence_type
                )
                for v in tree
            )
        )


def map_leaves(
    tree: Tree,
    f: Callable[[Leaf], Leaf],
    is_leaf: IsLeaf = is_leaf_default,
    *,
    dict_type: type = dict,
    sequence_type: type = tuple,
) -> Tree:
    """Apply `f` for each leaf in the tree."""
    if is_leaf(tree):
        return f(tree)
    elif isinstance(tree, MappingABC):
        return dict_type(
            (
                k,
                map_leaves(
                    v, f, is_leaf, dict_type=dict_type, sequence_type=sequence_type
                ),
            )
            for k, v in tree.items()
        )
    else:
        return sequence_type(
            map_leaves(v, f, is_leaf, dict_type=dict_type, sequence_type=sequence_type)
            for v in tree
        )


def flatten(tree: Tree, is_leaf: IsLeaf = is_leaf_default) -> Sequence[Leaf]:
    """Get the leaves of the tree."""
    return [x for x in iterate_leaves(tree, is_leaf)]


def flatten_with_paths(
    tree: Tree, is_leaf: IsLeaf = is_leaf_default
) -> Sequence[tuple[Path, Leaf]]:
    """Flatten a tree structure and return a list of tuples consisting of the path in
    the tree and the corresponding leaf.

    E.g.
    ```
    flatten_with_paths(
        {
            "a": [1, 2],
            "b": 3
        }
    )
    ```
    results in
    ```
    [
        (("a", 0), 1),
        (("a", 1), 2),
        (("b"), 3),
    ]
    ```
    """
    return [x for x in iterate_leaves_with_paths(tree, is_leaf)]


def iterate_leaves_with_paths(
    tree: Tree, is_leaf: IsLeaf = is_leaf_default, path_prefix: Path = tuple()
) -> Iterable[tuple[Path, Leaf]]:
    if is_leaf(tree):
        yield path_prefix, tree
    elif isinstance(tree, MappingABC):
        for k, v in tree.items():
            yield from iterate_leaves_with_paths(v, is_leaf, path_prefix + (k,))
    else:
        for i, v in enumerate(tree):
            yield from iterate_leaves_with_paths(v, is_leaf, path_prefix + (i,))


def iterate_leaves(tree: Tree, is_leaf: IsLeaf = is_leaf_default) -> Iterable[Leaf]:
    if is_leaf(tree):
        yield tree
    elif isinstance(tree, MappingABC):
        for v in tree.values():
            yield from iterate_leaves(v, is_leaf)
    else:
        for v in tree:
            yield from iterate_leaves(v, is_leaf)


_initial_missing = object()


def reduce_horizontal(
    fn: Callable[[Any, Leaf], Any],
    trees: Sequence[Tree],
    initial: Tree = _initial_missing,
    is_leaf: IsLeaf = is_leaf_default,
    dict_type: type = dict,
    sequence_type: type = tuple,
) -> Tree:
    """Reduce a list of trees such that elements with the same path in the tree are
    aligned.

    E.g.
    ```
    trees = [
        {"a": "a1", "b": ["b1"]},
        {"a": "a2", "b": ["b2"]}
    ]
    reduce_horizontal(str.__add__, trees)
    ```

    results in

    ```
    {"a": "a1a2", "b": ["b1b2"]}
    ```
    """
    if initial is _initial_missing:
        initial = trees[0]
        trees = trees[1:]

    return _reduce_horizontal(fn, trees, initial, is_leaf, dict_type, sequence_type)


def _reduce_horizontal(
    fn: Callable[[Any, Leaf], Any],
    trees: Sequence[Tree],
    initial: Tree,
    is_leaf: IsLeaf,
    dict_type: type,
    sequence_type: type,
) -> Tree:
    if len(trees) == 0:
        return initial

    if is_leaf(trees[0]):
        return functools.reduce(fn, trees, initial)

    if isinstance(trees[0], MappingABC):
        return dict_type(
            (
                k,
                _reduce_horizontal(
                    fn,
                    trees=[tree[k] for tree in trees],
                    initial=initial_v,
                    is_leaf=is_leaf,
                    dict_type=dict_type,
                    sequence_type=sequence_type,
                ),
            )
            for k, initial_v in initial.items()
        )
    return sequence_type(
        _reduce_horizontal(
            fn,
            trees=[tree[i] for tree in trees],
            initial=initial_v,
            is_leaf=is_leaf,
            dict_type=dict_type,
            sequence_type=sequence_type,
        )
        for i, initial_v in enumerate(initial)
    )
