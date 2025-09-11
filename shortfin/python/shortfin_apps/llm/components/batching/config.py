# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Provides types and classes to control batching behavior.
The idea is to encode all the knobs and control points into
a dataclass, and pass that to an opaque factory. To the caller
the internal API is completely opaque.
This prevents leakage of internal types and execution, and, consequently,
coupling of any kind, allowing us to modify and optimize as we want without
touching other code.
"""

import shortfin as sf

from enum import Enum
from dataclasses import dataclass

from ..config_struct import ModelParams


class BatchMode(Enum):
    DEFAULT = "Default"


@dataclass(slots=True)
class BatchConfig:
    mode: BatchMode
    model_params: ModelParams
    prefill_functions: dict[int, sf.ProgramFunction]  # type: ignore
    decode_functions: dict[int, sf.ProgramFunction]  # type: ignore
    prog_isolation: sf.ProgramIsolation  # type: ignore
    use_chunked_prefill: bool
    chunk_size: int = 1024
