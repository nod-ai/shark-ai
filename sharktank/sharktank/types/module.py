# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InferenceModule(Protocol):
    """Protocol for inference modules (both torch and IREE).

    This defines a common interface that both torch.nn.Module and
    TorchLikeIreeModule can satisfy, allowing them to be used
    interchangeably in inference code.

    Example:
        >>> def run_inference(model: InferenceModule, inputs):
        ...     return model(inputs)
        >>>
        >>> # Works with torch modules
        >>> torch_model = MyTorchModel()
        >>> run_inference(torch_model, x)
        >>>
        >>> # Also works with IREE modules
        >>> iree_model = adapt_torch_module_to_iree(torch_model, ...)
        >>> run_inference(iree_model, x)
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the module's forward pass."""
        ...

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the module's forward pass explicitly."""
        ...