# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tools for debugging models."""
from typing import Dict, Optional

from dataclasses import dataclass
import re
import os
from pathlib import Path
from typing import Sequence

import torch

from .logging import get_logger

__all__ = []

logger = get_logger("sharktank.debugging")

FLAGS_ENV_NAME = "TURBINE_LLM_DEBUG"
SETTING_PART_PATTERN = re.compile(r"""^([\\+\\-])?([^=]+)(=(.*))?$""")


@dataclass
class DebugFlags:
    enable_tensor_trace: bool = False
    enable_nan_checks: bool = False
    save_goldens_path: Optional[Path] = None
    golden_sequence_value: int = 0

    # Feature flags.
    # Enables use of custom IREE kernels in lieu of PyTorch general
    # for certain low level operations. We'd like to remove this flag but
    # certain eager use cases are still having problems with these custom
    # kernels, so keeping it to unblock progress.
    use_custom_iree_kernels: bool = True
    use_custom_generic_attention: bool = False

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
        elif name == "save_goldens_path":
            self.save_goldens_path = Path(value)
        elif name == "use_custom_iree_kernels":
            self.use_custom_iree_kernels = logical_sense
        elif name == "use_custom_generic_attention":
            self.use_custom_generic_attention = logical_sense
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
    key: str, t: torch.Tensor, *, values: bool = True, golden: bool = False
):
    trace_tensors(key, {"default": t}, values=values, golden=golden)


def trace_tensors(
    key: str,
    tensors: Dict[str, torch.Tensor],
    *,
    values: bool = True,
    golden: bool = False,
):
    if golden:
        if flags.save_goldens_path:
            _save_goldens(key, tensors)
        return
    if not flags.enable_tensor_trace:
        return
    for name, t in tensors.items():
        if t is not None:
            values_repr = repr(t) if values else "...elided..."
            print(f"::: TRACE {key}:{name}({list(t.shape), t.dtype}) =\n{values_repr}")


def _save_goldens(key: str, tensors: Dict[str, torch.Tensor]):
    next_sequence = flags.golden_sequence_value
    flags.golden_sequence_value += 1
    # Sanitize as path.
    key = re.sub("[" + re.escape(r"""#~!@$%^&*()[]{}:;"'""") + "]", "", key)
    from safetensors.torch import save_file

    path: Path = flags.save_goldens_path / f"{next_sequence:04d}_{key}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"::: SAVE GOLDEN {path}")
    non_none_tensors = {k: v.contiguous() for k, v in tensors.items() if v is not None}
    save_file(non_none_tensors, path)
