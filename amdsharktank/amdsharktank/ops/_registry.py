# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Signatures for dynamic dispatch of ops covering our fundamental tensor types."""

from collections.abc import Iterable
from inspect import isclass
from typing import Any, Callable, Iterable, Optional, Tuple

import collections
import functools
import inspect

from torch import Tensor
from amdsharktank.types import (
    PrimitiveTensor,
    QuantizedTensor,
    ReplicatedTensor,
    SplitPrimitiveTensor,
)

__all__ = [
    "AllOfExprs",
    "AllOfExprsVariadic",
    "AllOfType",
    "AnyOfType",
    "AnyType",
    "BoolTypeExprConst",
    "IsOfType",
    "AllNotOfType",
    "overridable",
    "SignatureDispatcher",
    "BoolTypeExpr",
    "get_all_registered_ops",
]

_TargetOverride = collections.namedtuple(
    "_TargetOverride",
    "salience, target, type_spec, auto_unbox, auto_dequant",
)


# Global registry of all registered operations
_GLOBAL_OP_REGISTRY: dict[str, "SignatureDispatcher"] = {}


def get_all_registered_ops() -> dict[str, "SignatureDispatcher"]:
    """Get a dictionary of all registered operations."""
    return _GLOBAL_OP_REGISTRY.copy()


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


def _matches(t, required):
    return isinstance(t, required) or (isinstance(t, type) and issubclass(t, required))


class BoolTypeExpr:
    """Expression that returns bool and accepts types as arguments."""

    def __init__(self, expr: Callable[..., bool]):
        self._expr = expr

    def __call__(self, *args: type) -> bool:
        return self._expr(*args)


class BoolTypeExprConst(BoolTypeExpr):
    """Anyways True."""

    def __init__(self, const: bool):
        def expr(*types: type):
            return const

        super().__init__(expr)


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
                any([_matches(t, required) for required in self._types]) for t in types
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
                [_matches(t, required) for t in types for required in self._types]
            )

        super().__init__(expr)


class AllNotOfType(BoolTypeExpr):
    """Returns True if none of the types are from a set of types.

    ```python
    # False. int is in (int, float).
    AllNotOfType(int, float)(int, str)

     # True. str is not in (int, float).
    AllNotOfType(int, float)(str, str)
    ```
    """

    def __init__(self, *types: type):
        self._types = types

        def expr(*types: type):
            return not any(
                [_matches(t, required) for t in types for required in self._types]
            )

        super().__init__(expr)


IsOfType = AllOfType


class AnyType:
    """Sentinel type that matches any type in override specifications.

    Use this when you want an override to match any type for a particular argument position.

    Example:
        @op.override(Tensor, Tensor, Tensor, AnyType)
        def my_override(a, b, c, d):
            # This will match when first 3 args are Tensors and 4th is anything
            ...
    """

    pass


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

    def __init__(self, sigf: Callable, is_trivially_replicable: bool = True):
        self._target_cache = dict()
        self._trampoline: Optional[Callable] = None
        self._overrides: list[_TargetOverride] = []
        self.is_trivially_replicable = is_trivially_replicable

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
        impl_name: str | None = None,
    ):
        def decorator(f):
            if f.__name__ == "_":
                f.__name__ = f"{self.__name__}__override"
            f._impl_name = impl_name
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

    def fail(self, tensors: tuple[Any, ...], impl_selection: str | None = None):
        spec = [type(t) for t in tensors]
        impl_msg = f" with impl selection '{impl_selection}'" if impl_selection else ""
        raise NotImplementedError(
            f"Overridable operator {self.__module__}.{self.__qualname__} does not "
            f"have an implementation for argument types: "
            f"{spec}{impl_msg}"
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
                if expected is AnyType:
                    continue
                if _matches(actual, expected):
                    continue
                # We expect kernels which are parameterized on Tensor to
                # unbox things that are isomorphic to it.
                is_expected_tensor = _matches(expected, Tensor)
                if is_expected_tensor:
                    if override.auto_unbox and _matches(actual, PrimitiveTensor):
                        continue
                    # Similarly, we conditionally allow auto dequant.
                    if override.auto_dequant and _matches(actual, QuantizedTensor):
                        continue
                break
            else:
                targets.append(override.target)
        return targets


def overridable(
    f: Callable[..., Any] | None = None,
    *,
    dispatch_args: Iterable[int | str] | None = None,
    is_trivially_replicable: bool = True,
):
    """Decorator to apply to overridable ops.

    Such ops can then have specializations stacked against them with the
    @override decorator.

    Parameters
    ----------
    dispatch_args:
        List of arguments to dispatch on. Can be name or index.

        If this is given a default trampoline method is created.
        The order matters for the dispatch resolution. The dispatch value list preserves
        the order given.
        If an argument is a variadic positional argument, its values are appended to
        the dispatch list.
        If an argument is a variadic keyword argument, its dictionary values are
        appended to the dispatch list.

        E.g.
        ```
        @overridable(dispatch_args=[1, a, c])
        def f(*a, b, c, d):
            ...

        f("a1", "a2", b="b", c="c", d="d")
        ```

        In this call the trampoline dispatch values would be
        `("b", "a1", "a2", "c")`.

    is_trivially_replicable:
        If True will automatically register a wrapper variant with all tensor
        arguments and results as replicated. This replicated op variant will call the
        underlying unreplicated variant with for all shards correspondingly. Then
        construct replicated results from all corresponding shards.
    """
    if f is None:
        return functools.partial(
            overridable,
            dispatch_args=dispatch_args,
            is_trivially_replicable=is_trivially_replicable,
        )

    dispatcher = SignatureDispatcher(f, is_trivially_replicable=is_trivially_replicable)
    functools.update_wrapper(dispatcher, f)

    if dispatch_args is not None:
        dispatcher.trampoline(make_default_trampoline(f, dispatch_args=dispatch_args))

    # Register the operation in the global registry
    op_name = f.__name__
    _GLOBAL_OP_REGISTRY[op_name] = dispatcher

    return dispatcher


def _parse_impl_selections(impl_selection: str | None) -> list[str]:
    """Parse implementation selection string with semicolon separator.

    Examples:
        "amdsharktank.asm" -> ["amdsharktank.asm"]
        "amdsharktank.asm;*" -> ["amdsharktank.asm", "*"]
        "amdsharktank.wave;amdsharktank.asm;*" -> ["amdsharktank.wave", "amdsharktank.asm", "*"]
    """
    if impl_selection is None:
        return ["*"]  # Default behavior selects any kernel
    if ";" in impl_selection:
        return impl_selection.split(";")
    return [impl_selection]


def _matches_impl_selection(impl_name: str | None, selection: str) -> bool:
    """Check if impl_name matches the given selection using hierarchical matching.

    Matches are done segment by segment, split by dots:
    - "amdsharktank" matches "amdsharktank", "amdsharktank.wave", "amdsharktank.asm"
    - "amdsharktank.wave" matches "amdsharktank.wave" but not "amdsharktank.asm"
    - "amdsharktank.wavelet" does not match "amdsharktank.wave"

    Args:
        impl_name: The _impl_name attribute from the override
        selection: A selection string, "*" matches anything

    Returns:
        True if impl_name matches the selection
    """
    if selection == "*":
        return True
    if impl_name is None:
        raise LookupError(
            "A kernel selection was specified and an implementation gave no implementation name"
        )
    # Split both into hierarchical segments
    selection_segments = selection.split(".")
    impl_segments = impl_name.split(".")
    # Selection must not have more segments than impl_name
    if len(selection_segments) > len(impl_segments):
        return False
    # Each selection segment must exactly match the corresponding impl segment
    for sel_seg, impl_seg in zip(selection_segments, impl_segments):
        if sel_seg != impl_seg:
            return False

    return True


def make_default_trampoline(
    f: Callable[..., Any], /, *, dispatch_args: Iterable[int | str]
) -> Callable[..., Any]:
    signature = inspect.signature(f)
    signature_arg_names = list(signature.parameters.keys())
    dispatch_args = [
        a if isinstance(a, str) else signature_arg_names[a] for a in dispatch_args
    ]

    def trampoline(_signature_dispatcher_: SignatureDispatcher, *args, **kwargs) -> Any:
        # We need the signature created here and not captured from the parent scope.
        # Otherwise torch tracing fails.
        impl_selection_str = kwargs.pop("impl", None)

        signature = inspect.signature(f)
        bound_args = signature.bind(*args, **kwargs)

        # Workaround for PyTorch versions < 2.7.1 where apply_defaults() doesn't work
        # correctly during tracing. Manually add missing default values.
        for param_name, param in signature.parameters.items():
            if (
                param_name not in bound_args.arguments
                and param.default is not inspect.Parameter.empty
            ):
                bound_args.arguments[param_name] = param.default

        dispatch_arg_values = []
        for dispatch_arg in dispatch_args:
            arg_value = bound_args.arguments[dispatch_arg]

            if (
                signature.parameters[dispatch_arg].kind
                == inspect.Parameter.VAR_POSITIONAL
            ):
                dispatch_arg_values.extend(arg_value)
            elif (
                signature.parameters[dispatch_arg].kind == inspect.Parameter.VAR_KEYWORD
            ):
                dispatch_arg_values.extend(arg_value.values())
            else:
                dispatch_arg_values.append(arg_value)

        # Implementation selection logic with preference support
        impl_selections = _parse_impl_selections(impl_selection_str)

        for impl_selection in impl_selections:
            for override in _signature_dispatcher_.find_overrides(dispatch_arg_values):
                # TODO: Remove this workaround - sharded operations need impl parameter
                # for recursive calls to non-sharded implementations
                call_kwargs = bound_args.kwargs.copy()
                has_sharded_args = any(
                    isinstance(arg, (ReplicatedTensor, SplitPrimitiveTensor))
                    for arg in dispatch_arg_values
                )
                if impl_selection_str is not None and has_sharded_args:
                    call_kwargs["impl"] = impl_selection_str

                impl_name = getattr(override, "_impl_name", None)
                if not has_sharded_args and not _matches_impl_selection(
                    impl_name, impl_selection
                ):
                    continue

                result = override(*bound_args.args, **call_kwargs)
                if result is not NotImplemented:
                    return override, result
        else:
            _signature_dispatcher_.fail(dispatch_arg_values, impl_selection)

    return trampoline
