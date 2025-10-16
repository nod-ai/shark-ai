# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Protocol-based interface for unified inference modules (torch and IREE)."""

from typing import Any, Protocol, runtime_checkable, Union, List
import torch
import iree.runtime


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
        >>> iree_model = load_torch_module_as_iree(torch_model, ...)
        >>> run_inference(iree_model, x)
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the module's forward pass."""
        ...

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the module's forward pass explicitly."""
        ...


def is_inference_module(obj: Any) -> bool:
    """Check if an object satisfies the InferenceModule protocol.

    Args:
        obj: Object to check

    Returns:
        True if the object has both __call__ and forward methods
    """
    return isinstance(obj, InferenceModule)


def get_inference_device(
    module: InferenceModule,
) -> Union[torch.device, iree.runtime.HalDevice]:
    """Get the device a module is running on.

    Args:
        module: An inference module (torch or IREE)

    Returns:
        The device the module is on (torch.device for torch modules,
        iree.runtime.HalDevice for IREE modules)

    Example:
        >>> torch_model = MyTorchModel().to("cuda")
        >>> get_inference_device(torch_model)  # torch.device('cuda:0')
        >>>
        >>> iree_model = load_torch_module_as_iree(torch_model, device="local-task")
        >>> get_inference_device(iree_model)  # <HalDevice local-task>
    """
    if isinstance(module, torch.nn.Module):
        # For torch modules, get device from first parameter
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters, assume CPU
            return torch.device("cpu")
    elif hasattr(module, "devices"):
        # For IREE modules (TorchLikeIreeModule)
        return module.devices[0]
    else:
        raise TypeError(f"Unknown module type: {type(module)}")


def inference_module_call(
    module: InferenceModule,
    *args: Any,
    method: str = "forward",
    **kwargs: Any,
) -> Any:
    """Call a method on an inference module in a unified way.

    This handles the slight differences between torch and IREE modules
    when calling methods other than forward().

    Args:
        module: The inference module to call
        *args: Positional arguments
        method: Name of the method to call (default: "forward")
        **kwargs: Keyword arguments

    Returns:
        The module's output

    Example:
        >>> # Call forward
        >>> output = inference_module_call(model, input_tensor)
        >>>
        >>> # Call a custom method
        >>> output = inference_module_call(model, x, method="generate")
    """
    method_fn = getattr(module, method)
    return method_fn(*args, **kwargs)
