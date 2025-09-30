# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_compile_flags = [
    "--iree-execution-model=async-external",
    "--iree-global-opt-propagate-transposes=1",
    "--iree-opt-const-eval=0",
    "--iree-opt-outer-dim-concat=1",
    "--iree-opt-aggressively-propagate-transposes=1",
    "--iree-codegen-llvmgpu-use-vector-distribution=1",
    "--iree-llvmgpu-enable-prefetch=1",
    "--iree-opt-data-tiling=0",
    "--iree-vm-target-truncate-unsupported-floats",
    "--iree-dispatch-creation-enable-aggressive-fusion",
    "--iree-hal-memoization=1",
    "--iree-codegen-llvmgpu-use-tile-and-fuse-matmul=1",
    "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize),iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-dispatch-creation-bubble-up-expand-shapes, canonicalize, cse, canonicalize), util.func(iree-preprocessing-generalize-linalg-matmul-experimental))",
]
