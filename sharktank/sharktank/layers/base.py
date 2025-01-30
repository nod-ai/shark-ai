# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, Optional
from collections import OrderedDict
from collections.abc import Mapping
import torch
import torch.nn as nn

from ..types import InferenceTensor, Theta, AnyTensor
from ..utils import debugging
from .. import ops

__all__ = [
    "BaseLayer",
    "ThetaLayer",
]


def _set_recursively_submodules_default_trace_tensor_key_prefix(
    module: nn.Module, prefix: str = ""
):
    if isinstance(module, BaseLayer):
        module.trace_tensor_key_prefix = prefix

    for name, submodule in module.named_children():
        submodule_prefix = f"{prefix}{name}."
        _set_recursively_submodules_default_trace_tensor_key_prefix(
            submodule, submodule_prefix
        )


class BaseLayer(nn.Module):
    """Base class of all of our layers."""

    def __init__(self):
        super().__init__()
        self._trace_tensor_key_prefix = ""

    def set_recursively_submodules_default_trace_tensor_key_prefix(self):
        """All submodules get a trace key prefix that reflects their nesting with
        respect to the parent module.

        Example:
        ```
        class A(BaseLayer):
            def __init__(self):
                ...
                self.b = ...

        class B(BaseLayer):
            def __init__(self):
                ...
                self.c = ...

        class C(BaseLayer):
            def forward(self, x):
                self.trace_tensor("x", x)


        a = A()
        a.set_recursively_submodules_default_trace_tensor_key_prefix()
        ```

        This will result in trace key prefixes
        a -> ""
        a.b -> "b."
        a.b.c -> "b.c."

        The trace_tensor method call in C.forward will result in a trace with key
        "b.c.x".
        """
        _set_recursively_submodules_default_trace_tensor_key_prefix(
            self, self.trace_tensor_key_prefix
        )

    @property
    def trace_tensor_key_prefix(self) -> str:
        """When tracing with self.trace_tensor all keys will be prefixed by this
        string.
        The default prefix is the empty string."""
        return self._trace_tensor_key_prefix

    @trace_tensor_key_prefix.setter
    def trace_tensor_key_prefix(self, value: str):
        self._trace_tensor_key_prefix = value

    def trace_tensor(
        self,
        key: str,
        tensors: Dict[str, torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    ):
        debugging.trace_tensor(f"{self.trace_tensor_key_prefix}{key}", tensors)

    def assert_not_nan(self, *ts: torch.Tensor):
        """Checks whether tensors have nan values in them.

        Must be enabled via a global switch as this kind of checking is not
        accelerator or compilation friendly.
        """
        if debugging.flags.enable_nan_checks:
            for t in ts:
                if torch.isnan(t).any():
                    raise AssertionError(f"Tensor contains nans! {t}")

    def sample_inputs(
        self, batch_size: int = 1, function: Optional[str] = None
    ) -> tuple[tuple[AnyTensor], OrderedDict[str, AnyTensor]]:
        """Return sample inputs that can be used to run the function from the model.
        If function is None then layer is treated as the callable.
        E.g.
        ```
        args, kwargs = model.sample_inputs()
        model(*args, **kwargs)
        ```

        One purpose of this method is to standardize exportation of models to MLIR.
        """
        raise NotImplementedError()


class ThetaLayer(BaseLayer):
    "Base class for layers that derive parameters from a Theta object."

    def __init__(self, theta: Theta):
        super().__init__()
        self.theta = theta

    def theta_tensor(self, name: str) -> InferenceTensor:
        # TODO: We may need to do some bookkeeping here to ensure export
        # tracks all of these.
        return self.theta.tensor(name)
