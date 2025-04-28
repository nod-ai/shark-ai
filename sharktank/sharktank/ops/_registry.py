# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from typing import Any, Callable, Iterable, Optional, Tuple

import collections
import functools

from torch import Tensor
from sharktank.types import PrimitiveTensor, QuantizedTensor

__all__ = [
    "AllOfExprs",
    "AllOfExprsVariadic",
    "AllOfType",
    "AnyOfType",
    "IsOfType",
    "NotOfType",
    "overridable",
    "SignatureDispatcher",
    "BoolTypeExpr",
]

_TargetOverride = collections.namedtuple(
    "_TargetOverride",
    "salience, target, type_spec, auto_unbox, auto_dequant",
)


# When an op is dispatched, it will be stashed here for testing to verify.
# Use _test_enable_last_op_dispatch(True) / _test_enable_last_op_dispatch(False)
# in test cases to enable/disable tracking of the last op dispatched.
# The last op can be queried with _test_get_last_op_dispatch().
_ENABLE_TEST_LAST_OP_DISPATCH = False
_TEST_LAST_OP_DISPATCH = None


def _test_enable_last_op_dispatch(en: bool = True):
    global _TEST_LAST_OP_DISPATCH
    global _ENABLE_TEST_LAST_OP_DISPATCH
    _TEST_LAST_OP_DISPATCH = None
    _ENABLE_TEST_LAST_OP_DISPATCH = en


def _test_get_last_op_dispatch():
    assert (
        _ENABLE_TEST_LAST_OP_DISPATCH
    ), "Cannot get last op dispatched without calling _test_enable_last_op_dispatch()"
    return _TEST_LAST_OP_DISPATCH


class BoolTypeExpr:
    """Expression that returns bool and accepts types as arguments."""

    def __init__(self, expr: Callable[..., bool]):
        self._expr = expr

    def __call__(self, *args: type) -> bool:
        return self._expr(*args)


class AllOfExprs(BoolTypeExpr):
    """Returns True if all type arguments match their respective boolean type
    expression.

    ```python
    # True. int == int and str in (float, str).
    AllOfExprs(IsOfType(int), IsOfType(float, str))(int, str)

     # False. str is not in (int, float).
    AllOfExprs(IsOfType(int), IsOfType(int, float))(int, str)
    ```
    """

    def __init__(self, *exprs: BoolTypeExpr):
        self._exprs = exprs

        def expr(*types: type):
            if len(types) < len(self._exprs):
                return False
            return all([e(t) for e, t in zip(self._exprs, types)])

        super().__init__(expr)


class AllOfExprsVariadic(BoolTypeExpr):
    """Returns True if all type arguments match their respective boolean type
    expression and any remaining trailing arguments match the last type expression.

    ```python
    # True. int == int
    # str in (float, str).
    # float in (float, str).
    AllOfExprsVariadic(IsOfType(int), IsOfType(float, str))(int, str, float)

     # False. str is not in (int, float).
    AllOfExprsVariadic(IsOfType(int), IsOfType(int, float))(int, float, str, int)
    ```
    """

    def __init__(self, *exprs: BoolTypeExpr):
        if len(exprs) == 0:
            raise ValueError("At least one expression is required.")
        self._exprs = list(exprs)

        def expr(*types: type):
            if len(types) < len(self._exprs):
                return False
            exprs = self._exprs
            if len(types) > len(exprs):
                # pad with the trailing expression.
                exprs = exprs + ([exprs[-1]] * (len(types) - len(self._exprs)))
            return all([e(t) for e, t in zip(exprs, types)])

        super().__init__(expr)


class AllOfType(BoolTypeExpr):
    """Returns True if all of the types are from a set of types.

    ```python
    # False. str is not in (int, float).
    AllOfType(int, float)(int, str)

     # True. int is in (int, float).
    AllOfType(int, float)(int, int)
    ```
    """

    def __init__(self, *types: type):
        self._types = types

        def expr(*types: type):
            return all(
                any([issubclass(t, required) for required in self._types])
                for t in types
            )

        super().__init__(expr)


class AnyOfType(BoolTypeExpr):
    """Returns True if any of the types are from a set of types.

    ```python
    # True. int is in (int, float).
    AnyOfType(int, float)(int, str)

     # False. str is not in (int, float).
    AnyOfType(int, float)(str, str)
    ```
    """

    def __init__(self, *types: type):
        self._types = types

        def expr(*types: type):
            return any(
                [issubclass(t, required) for t in types for required in self._types]
            )

        super().__init__(expr)


class NotOfType(BoolTypeExpr):
    """Returns True if none of the types are from a set of types.

    ```python
    # False. int is in (int, float).
    NotOfType(int, float)(int, str)

     # True. str is not in (int, float).
    NotOfType(int, float)(str, str)
    ```
    """

    def __init__(self, *types: type):
        self._types = types

        def expr(*types: type):
            return not any(
                [issubclass(t, required) for t in types for required in self._types]
            )

        super().__init__(expr)


IsOfType = AllOfType


class SignatureDispatcher:
    """Replaces an overridable function with a tensor type base dispatcher.

    When overrides are registered, the computed target cache is cleared but
    between registrations, it is maintained for quick lookup by a tuple of
    tensor types in the order of the formal tensor arguments of the original
    function signature.
    """

    __slot__ = [
        # "_sig",
        # "_tensor_names",
        "_overrides",
        "_target_cache",
        "_trampoline",
    ]

    def __init__(self, sigf: Callable):
        self._target_cache = dict()
        self._trampoline: Optional[Callable] = None
        self._overrides: list[_TargetOverride] = []

    def __call__(self, *args, **kwargs):
        trampoline = self._trampoline
        assert trampoline is not None
        selected_override, *results = trampoline(self, *args, **kwargs)
        if _ENABLE_TEST_LAST_OP_DISPATCH:
            global _TEST_LAST_OP_DISPATCH
            _TEST_LAST_OP_DISPATCH = selected_override
        arity = len(results)
        if arity == 1:
            return results[0]
        elif arity == 0:
            return None
        else:
            return results

    def override(
        self,
        *type_spec: tuple[type | BoolTypeExpr, ...],
        salience: int = 0,
        auto_unbox: bool = True,
        auto_dequant: bool = False,
    ):
        def decorator(f):
            if f.__name__ == "_":
                f.__name__ = f"{self.__name__}__override"
            self._overrides.append(
                _TargetOverride(
                    salience=salience,
                    target=f,
                    type_spec=type_spec,
                    auto_unbox=auto_unbox,
                    auto_dequant=auto_dequant,
                )
            )
            self._overrides.sort(key=lambda v: v.salience)
            self._target_cache.clear()  # Need to recompute all targets
            return f

        return decorator

    def find_overrides(self, tensors: tuple[Any, ...]) -> Iterable[Callable]:
        """Finds the most salient override for the given named tensors."""
        type_spec = tuple(type(t) for t in tensors)
        found_targets = self._target_cache.get(type_spec)
        if found_targets is None:
            # Slow-path try to find it.
            found_targets = self._match_targets(type_spec)
            self._target_cache[type_spec] = found_targets
        return reversed(found_targets)

    def fail(self, tensors: tuple[Any, ...]):
        spec = [type(t) for t in tensors]
        raise NotImplementedError(
            f"Overridable operator {self.__module__}.{self.__qualname__} does not "
            f"have an implementation for argument types: "
            f"{spec}"
        )

    def get_override_names(self):
        return [o.target.__name__ for o in self._overrides]

    def remove_override(self, override_name: str):
        self._overrides = [
            o for o in self._overrides if o.target.__name__ != override_name
        ]
        self._target_cache.clear()

    def trampoline(self, trampoline: Callable):
        assert self._trampoline is None
        self._trampoline = trampoline

    def _is_type_expr_target(
        self, override_type_spec: Tuple[type, ...], type_spec: Tuple[type, ...]
    ) -> bool:
        if len(override_type_spec) > 0 and isinstance(
            override_type_spec[0], BoolTypeExpr
        ):
            if len(override_type_spec) > 1:
                raise TypeError(
                    f"Override with multiple arguments not allowed when using BoolTypeExpr. Type spec: {override_type_spec}"
                )
            return True
        return False

    def _is_type_expr_target_match(
        self, type_expr: BoolTypeExpr, type_spec: Tuple[type, ...]
    ) -> bool:
        return type_expr(*type_spec)

    def _match_targets(self, type_spec: tuple):
        targets = []
        for override in self._overrides:
            override_type_spec = override.type_spec

            # Check if the override is a boolean type expression and if it is that it
            # satisfied.
            if self._is_type_expr_target(override_type_spec, type_spec):
                if self._is_type_expr_target_match(override_type_spec[0], type_spec):
                    targets.append(override.target)
                continue

            if len(override_type_spec) != len(type_spec):
                continue
            for expected, actual in zip(override.type_spec, type_spec):
                if expected is None:
                    continue
                if issubclass(actual, expected):
                    continue
                # We expect kernels which are parameterized on Tensor to
                # unbox things that are isomorphic to it.
                is_expected_tensor = issubclass(expected, Tensor)
                if is_expected_tensor:
                    if override.auto_unbox and issubclass(actual, PrimitiveTensor):
                        continue
                    # Similarly, we conditionally allow auto dequant.
                    if override.auto_dequant and issubclass(actual, QuantizedTensor):
                        continue
                break
            else:
                targets.append(override.target)
        return targets


def overridable(f):
    """Decorator to apply to overridable ops.

    Such ops can then have specializations stacked against them with the
    @override decorator.
    """
    dispatcher = SignatureDispatcher(f)
    functools.update_wrapper(dispatcher, f)
    return dispatcher
