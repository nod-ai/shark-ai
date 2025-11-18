# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tools for debugging models."""
from typing import Callable, Dict, Optional, Tuple
from collections.abc import Mapping
from dataclasses import dataclass
import re
import os
from pathlib import Path
import iree.turbine.support.debugging

import torch

from .logging import get_logger

__all__ = []

logger = get_logger(__name__)

FLAGS_ENV_NAME = "TURBINE_LLM_DEBUG"
SETTING_PART_PATTERN = re.compile(r"""^([\\+\\-])?([^=]+)(=(.*))?$""")


@dataclass
class DebugFlags:
    enable_tensor_trace: bool = False
    enable_nan_checks: bool = False
    trace_path: Optional[Path] = None
    """Directory location for tensor tracing.

    See: amdsharktank.utils.debugging.trace_tensor_to_safetensors_callback
    """

    # Feature flags.
    # Enables use of custom IREE kernels in lieu of PyTorch general
    # for certain low level operations. We'd like to remove this flag but
    # certain eager use cases are still having problems with these custom
    # kernels, so keeping it to unblock progress.
    use_custom_iree_kernels: bool = True

    def set(self, part: str):
        m = re.match(SETTING_PART_PATTERN, part)
        if not m:
            logger.warn("Syntax error in %s flag: '%s'", FLAGS_ENV_NAME, part)
            return
        logical_sense = m.group(1) != "-"
        name = m.group(2)
        value = m.group(4)

        if name == "tensor_trace":
            self.enable_tensor_trace = logical_sense
        elif name == "enable_nan_checks":
            self.enable_nan_checks = logical_sense
        elif name == "trace_path":
            self.trace_path = Path(value)
        elif name == "use_custom_iree_kernels":
            self.use_custom_iree_kernels = logical_sense
        else:
            logger.warn("Unrecognized %s flag: '%s'", FLAGS_ENV_NAME, name)

    @staticmethod
    def parse(settings: str) -> "DebugFlags":
        new_flags = DebugFlags()
        parts = settings.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            new_flags.set(part)
        return new_flags

    @staticmethod
    def parse_from_env() -> "DebugFlags":
        settings = os.getenv(FLAGS_ENV_NAME)
        if settings is None:
            return DebugFlags()
        new_flags = DebugFlags.parse(settings)
        logger.debug("Parsed debug flags from env %s: %r", FLAGS_ENV_NAME, new_flags)
        return new_flags


flags = DebugFlags.parse_from_env()


def trace_tensor(
    key: str, tensors: Dict[str, torch.Tensor] | list[torch.Tensor] | torch.Tensor
):
    if not flags.enable_tensor_trace:
        return

    if isinstance(tensors, Mapping):
        sub_keys = list(tensors.keys())
        sub_keys.sort()

        for sub_key in sub_keys:
            trace_tensor(f"{key}.{sub_key}", tensors[sub_key])
        return

    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)

    from amdsharktank import ops

    ops.trace_tensor(key, *tensors)


TraceKey = str
TraceTensors = Callable[[TraceKey, *Tuple[torch.Tensor, ...]], None]


def set_trace_tensor_callback(callback: TraceTensors):
    """Set callback for amdsharktank.ops.trace_tensor.

    This is what will be called when tensors are traced with the op."""
    iree.turbine.support.debugging.trace_tensor_callback = callback


def get_trace_tensor_callback() -> Optional[TraceTensors]:
    return iree.turbine.support.debugging.trace_tensor_callback


def null_trace_tensor_callback(key: str, *tensors: Tuple[torch.Tensor]):
    return


def trace_tensor_to_safetensors_callback(key: str, *tensors: Tuple[torch.Tensor]):
    """A trace sink that will record tensors to safetensors files.

    Use amdsharktank.utils.debugging.flags.trace_path to set the directory location.
    """
    if len(tensors) == 1:
        tensors_in_dict = {"": t for t in tensors}
    else:
        tensors_in_dict = {f"{i}": t for i, t in enumerate(tensors)}
    trace_tensors_to_safetensors(key, tensors_in_dict)


set_trace_tensor_callback(trace_tensor_to_safetensors_callback)


def trace_tensors_to_safetensors(key: str, tensors: Dict[str, torch.Tensor]):
    # Sanitize as path.
    key = re.sub("[" + re.escape(r"""*?"<>|""") + "]", "", key)
    from safetensors.torch import save_file

    path: Path = flags.trace_path / f"{key}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"::: TRACE TENSOR(S) {path}")
    non_none_tensors = {k: v.contiguous() for k, v in tensors.items() if v is not None}
    save_file(non_none_tensors, filename=path)
