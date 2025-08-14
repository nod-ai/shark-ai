# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration system for op comparison tests."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional


@dataclass
class OpTestConfig:
    """Configuration for testing op implementations.

    Attributes:
        op: The op from sharktank.ops (e.g., ops.scaled_dot_product_attention)
        reference_impl: Direct function reference to the reference implementation
        test_impls: List of implementations to test. If None, auto-discover all.
        args: List of arguments to pass to the op (tensors or None for optional args)
        kwargs: Additional keyword arguments to pass to the op
        rtol: Relative tolerance for numeric comparison
        atol: Absolute tolerance for numeric comparison
        cosine_similarity_threshold: Threshold for cosine similarity comparison
        comparison_method: Method to use for comparison ("assert_close", "cosine_similarity", or "both")
        fail_on_not_implemented: If True, fail test when implementation returns NotImplemented. If False, skip.
    """

    op: Callable  # The op from sharktank.ops
    reference_impl: Callable  # Direct function reference
    test_impls: Optional[List[Callable]] = None  # If None, auto-discover all
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    rtol: float = 1e-3
    atol: float = 1e-3
    cosine_similarity_threshold: float = 0.99
    comparison_method: str = "assert_close"  # or "cosine_similarity" or "both"
    fail_on_not_implemented: bool = (
        False  # If True, fail instead of skip for NotImplemented
    )
