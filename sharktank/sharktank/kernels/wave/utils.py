# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.ir import (
    Module,
    StringAttr,
)


def get_wave_module_body_asm(module: Module) -> str:
    """
    Concatenates the MLIR of all operations within the
    body region of the top-level wave_compile() module and modifies the
    visibility of the top-level public FuncOp generated in wave_compile()
    to private, so that it gets removed when inlined.
    """
    block = module.operation.regions[0].blocks[0]
    ops_asm = []
    for op in block.operations:
        if op.operation.name == "func.func":
            op.attributes["sym_visibility"] = StringAttr.get("private")
        ops_asm.append(op.get_asm())

    return "\n".join(ops_asm)
