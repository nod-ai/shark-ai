# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING, Union
from collections.abc import Mapping, Iterable
from amdsharktank.types import InferenceTensor, unbox_tensor
from amdsharktank.utils import verify_exactly_one_is_not_none
from amdsharktank.utils.logging import get_logger
import fnmatch
import os
import re
import torch

if TYPE_CHECKING:
    from amdsharktank.types import AnyTensor, InferenceTensor

logger = get_logger(__name__)


class FilterKind(Enum):
    INCLUDE = 1
    EXCLUDE = 2


@dataclass
class PatchFilterElement:
    regex: str | None = None
    fnmatch: str | None = None
    kind: FilterKind = FilterKind.INCLUDE

    def __post_init__(self):
        verify_exactly_one_is_not_none(regex=self.regex, fnmatch=self.fnmatch)


default_patch_filter = [PatchFilterElement(regex=".*\\.forward$")]


def is_filter_match(name: str, filter: list[PatchFilterElement]) -> bool:
    for f in filter:
        if f.regex is not None:
            is_match = re.match(f.regex, name)
        else:
            name_for_fnmatch = name.replace(".", os.sep)
            fnmatch_pattern = f.fnmatch.replace(".", os.sep)
            is_match = fnmatch.fnmatchcase(name_for_fnmatch, fnmatch_pattern)

        if not is_match:
            continue
        if f.kind == FilterKind.INCLUDE:
            return True
        if f.kind == FilterKind.EXCLUDE:
            return False
    return False


class Patch:
    """Patches calls to methods, allowing various forms of interception.

    Can patch the pre and post calling submodule methods method."""

    def patch_child_modules(
        self,
        module: torch.nn.Module,
        *,
        filter: list[PatchFilterElement] = default_patch_filter,
    ):
        """Wraps methods of a module and its child submodules.

        The main usage of this is to monkey-patch a model to insert tensor tracing
        without having to alter its source code.

        Different types of callbacks can be specified to control wrapping:
        * before_call: Called with (method_path, module, args, kwarg) before
        a method. Used for logging inputs to a module.
        * after_call: Called with (method_path, module, results) after the
        method returns. Used for logging results.

        The filter argument specifies what methods to patch based on their
        fully-qualified name.
        E.g.
        `[PatchFilterElement(regex=".*\\.forward$")]` will patch forward methods in all
        submodules. This is the default.
        The order of filter elements is important. All filter elements are matched
        one-by-one. The first successful match terminates the iteration.
        A successful match on filter element of kind PatchFilterElement.INCLUDE would cause
        the final match to succeeded.
        A successful match on filter element of PatchFilterElement.EXCLUDE inverts
        the meaning and would cause the final match to fail.
        If no filter element is a match then the method is rejected and is not patched.
        E.g.
        ```
        [
            PatchFilterElement(regex=".*\\.layer\.0\\.attention\\..+$", kind=FilterKind.EXCLUDE)
            PatchFilterElement(regex=".*\\.attention\\.forward")
        ]
        ```
        This would match `llm.layer.1.attention.forward`, but it would not match
        `llm.layer.0.attention.forward`.

        Another option is to use fnmatch-like matching
        ```
        PatchFilterElement(fnmatch="*.attention.forward")
        ```
        See https://docs.python.org/3/library/fnmatch.html

        If you are using Python >= 3.13 you could use the Python's standard module
        glob to translate a glob pattern into a regex.
        ```
        regex = glob.translate('**.attention.forward', recursive=True, include_hidden=True, seps=["."])
        PatchFilterElement(regex=regex)
        ```

        If regex name matching is not enough. Module methods can be patched directly with
        Patching.patch_method.
        ```
        my_patching.patch_method(
            method=my_module.forward,
            attribute_name="forward",
            name_prefix="my_module.",
            module=my_module,
        )
        ```
        """

        def _patch(name: str, m: torch.nn.Module):
            for attribute_name in dir(m):
                fully_qualified_name = f"{name}.{attribute_name}"
                if not is_filter_match(fully_qualified_name, filter):
                    continue
                attribute = getattr(m, attribute_name)
                if not callable(attribute):
                    continue
                if isinstance(attribute, torch.nn.Module):
                    # Avoid overriding torch modules as they are callables as well.
                    continue

                self.patch_method(
                    method=attribute,
                    attribute_name=attribute_name,
                    name_prefix=name,
                    module=m,
                )

        for name, m in module.named_modules():
            _patch(name, m)

    def before_call(
        self,
        method_path: str,
        module: torch.nn.Module,
        args: list[Any],
        kwargs: dict[str, Any],
    ):
        """Called before every patched method function.

        Args:
            method_path: Fully qualified submodule and method name.
            E.g. `model.submodule_a.forward`.
        """
        pass

    def after_call(self, method_path: str, module: torch.nn.Module, results):
        """Called after every patched method function with results."""
        ...

    def patch_method(
        self,
        method: Callable[..., Any],
        attribute_name: str,
        name_prefix: str,
        module: torch.nn.Module,
    ):
        name_prefix = f"{name_prefix}.{attribute_name}"

        def wrapper(*args, **kwargs):
            self.before_call(name_prefix, module, args, kwargs)
            results = method(*args, **kwargs)
            if not isinstance(results, tuple):
                results = (results,)
            self.after_call(name_prefix, module, results)
            return results

        setattr(module, attribute_name, wrapper)


class SaveModuleResultTensorsPatch(Patch):
    """Module patch which saves the args/results of all module calls to a safetensors
    file.

    Duplicate module invocations are suffixed with "#n" where n is the zero
    based call counter.

    Users must call save_file() once all tensors have been accumulated.
    """

    def __init__(self, with_before_call: bool = False):
        self.with_before_call = with_before_call
        self.tensors: dict[str, torch.Tensor] = {}
        # Map of tensor name to last used index for duplicated tensors.
        self.duplicate_tensors: dict[str, torch.Tensor] = {}

    def before_call(
        self,
        method_path: str,
        module: torch.nn.Module,
        args: list[Any],
        kwargs: dict[str, Any],
    ):
        if not self.with_before_call:
            return

        self._add_nested_tensors(
            name_prefix=f"{method_path}.arg", maybe_tensors=args, name_delimiter="%"
        )
        self._add_nested_tensors(
            name_prefix=f"{method_path}.arg", maybe_tensors=kwargs, name_delimiter="%"
        )

    def after_call(self, method_path: str, module: torch.nn.Module, results: Any):
        self._add_nested_tensors(
            name_prefix=method_path, maybe_tensors=results, name_delimiter="%"
        )

    def save_file(self, output_path: Path, *, skip_unsupported_dtypes: bool = False):
        """Saves accumulated tensors to the given file.
        Args:
        skip_unsupported_dtypes:
            skip tensors with dtype that is unsupported by safetensors.
            Warn when such a tensor is encountered."""
        from safetensors.torch import save_file

        tensor_dict = self.tensors
        if skip_unsupported_dtypes:
            safetensors_unsupported_dtypes = set(
                [torch.complex32, torch.complex64, torch.complex128]
            )
            unsupported_tensor_dict = {
                k: v
                for k, v in self.tensors.items()
                if v.dtype in safetensors_unsupported_dtypes
            }
            if len(unsupported_tensor_dict) > 0:
                unsupported_dtypes = {
                    k: v.dtype for k, v in unsupported_tensor_dict.items()
                }
                logger.warning(
                    f"Safetensors could not save tensor(s) with dtype {unsupported_dtypes}"
                )
                tensor_dict = {
                    k: v
                    for k, v in tensor_dict.items()
                    if k not in unsupported_tensor_dict.keys()
                }

        save_file(tensor_dict, output_path)

    def _add_nested_tensors(
        self,
        name_prefix: str,
        maybe_tensors: list[Any] | dict[str, Any] | torch.Tensor | Any,
        name_delimiter: str,
    ):
        if isinstance(maybe_tensors, str):
            return

        if isinstance(maybe_tensors, (torch.Tensor, InferenceTensor)):
            self._add_tensor(name=name_prefix, tensor=unbox_tensor(maybe_tensors))
        elif isinstance(maybe_tensors, Mapping):
            for k, v in maybe_tensors.items():
                self._add_nested_tensors(
                    f"{name_prefix}{name_delimiter}{k}", v, name_delimiter
                )
        elif isinstance(maybe_tensors, Iterable):
            for i, v in enumerate(maybe_tensors):
                self._add_nested_tensors(
                    f"{name_prefix}{name_delimiter}{i}", v, name_delimiter
                )
        else:
            logger.warning(f"Could not handle element of type {type(maybe_tensors)}.")

    def _add_tensor(self, name: str, tensor: torch.Tensor):
        tensor = torch.detach(tensor).contiguous().to(device="cpu").clone()
        if name in self.tensors:
            orig_dup = self.tensors[name]
            del self.tensors[name]
            self.duplicate_tensors[name] = 0
            self.tensors[f"{name}#0"] = orig_dup
        if name in self.duplicate_tensors:
            index = self.duplicate_tensors[name] + 1
            self.duplicate_tensors[name] = index
            self.tensors[f"{name}#{index}"] = tensor
        else:
            self.tensors[name] = tensor


class TraceTensorModulePatch(Patch):
    """Trace tensors using the amdsharktank.ops.trace_tensor mechanism.

    This can be used to trace tensors both in eager and during execution with IREE.
    Usually it allows to get adequate tracing density when models are decomposed into
    multiple nested torch modules.
    """

    def __init__(
        self, with_before_call: bool = False, exclude_regex: str | None = None
    ):
        """
        exclude_regex: exclude fully qualified trace keys that match a regex search
            with this pattern.
        """
        self.with_before_call = with_before_call
        self.exclude_regex = exclude_regex

    def before_call(
        self,
        method_path: str,
        module: torch.nn.Module,
        args: list[Any],
        kwargs: dict[str, Any],
    ):
        if not self.with_before_call:
            return

        self.trace_tensor(
            method_path=method_path,
            module=module,
            key="arg",
            args=args,
            kwargs=kwargs,
        )

    def after_call(self, method_path: str, module: torch.nn.Module, results: Any):
        self.trace_tensor(
            method_path=method_path,
            module=module,
            key="",
            args=results,
            kwargs={},
        )

    def trace_tensor(
        self,
        method_path: str,
        module: torch.nn.Module,
        key: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ):
        from amdsharktank.layers import BaseLayer
        from amdsharktank import ops

        def _trace_if_tensor(key: str, maybe_tensor: Union["AnyTensor", Any]):
            if self.exclude_regex is not None and re.search(
                self.exclude_regex, f"{method_path}.{key}"
            ):
                return
            if not isinstance(maybe_tensor, (torch.Tensor, InferenceTensor)):
                return

            ops.trace_tensor(f"{method_path}.{key}", maybe_tensor)

        if isinstance(module, BaseLayer):
            for i, arg in enumerate(args):
                _trace_if_tensor(key=f"{key}%{i}", maybe_tensor=arg)
            for arg_name, arg in kwargs.items():
                _trace_if_tensor(key=f"{key}%{arg_name}", maybe_tensor=arg)
