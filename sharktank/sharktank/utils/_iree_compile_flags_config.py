# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
IREE compilation flags for specific usecases.
"""

LLM_HIP_COMPILE_FLAGS = [
    "--iree-hal-target-device=hip",
    "--iree-hip-target=gfx942",  # MI300 example; adjust to your GPU if needed
    "--iree-execution-model=async-external",
    "--iree-opt-strip-assertions=true",
    "--iree-opt-level=O3",
    "--iree-dispatch-creation-propagate-collapse-across-expands=true",
    "--iree-stream-affinity-solver-max-iterations=1024",
    "--iree-hal-indirect-command-buffers=true",
    "--iree-stream-resource-memory-model=discrete",
    "--iree-hip-specialize-dispatches",
    "--iree-hal-memoization=true",
    "--iree-codegen-enable-default-tuning-specs=true",
]
