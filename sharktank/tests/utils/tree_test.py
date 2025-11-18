# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
from unittest import TestCase
from amdsharktank.utils.tree import flatten, map_leaves, reduce_horizontal
from amdsharktank.utils import iterables_equal


class TreeTest(TestCase):
    def testReduceHorizontal(self):
        trees = [{"a": "a1", "b": ["b1"]}, {"a": "a2", "b": ["b2"]}]
        result = reduce_horizontal(str.__add__, trees)
        expected = {"a": "a1a2", "b": ["b1b2"]}
        assert result["a"] == expected["a"]
        assert result["b"][0] == expected["b"][0]

    def testMapLeaves(self):
        tree = {"a": "a", "b": ["b"]}
        result = map_leaves(tree, str.upper)
        expected = {"a": "A", "b": ["B"]}
        assert result["a"] == expected["a"]
        assert result["b"][0] == expected["b"][0]

    def testFlatten(self):
        tree = [1, {"a": 2, "b": 3}, [4]]
        assert iterables_equal(flatten(tree), [1, 2, 3, 4])
