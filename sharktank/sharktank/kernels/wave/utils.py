# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.ir import (
    Module,
    StringAttr,
)
import re
import functools


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


# Disallowed characters in an MLIR suffix-id
_DISALLOWED = re.compile(r"[^A-Za-z0-9\$\._-]")


def mangle(base_name: str, **kwargs) -> str:
    r"""
    Build a readable, deterministic MLIR kernel name (note the double underscore
    after `base_name`):
    ```
    base_name__key1_val1_key2_val2_...
    ```
    Make sure the `kwargs` uniquely identify the kernel for any shapes or dtypes
    it can take. TODO: is this the right defn of unique?
    Keys are sorted so the output is stable.
    According to the MLIR LangRef, only characters matching the regex
    `[A-Za-z0-9\$\._-]` are allowed in an unquoted suffix-id. Any other
    characters are simply removed.
    """
    parts: list[str] = [base_name, ""]

    for key in kwargs:
        val = kwargs[key]
        parts.append(f"{str(key)}_{str(val)}")

    return re.sub(_DISALLOWED, "", "_".join(parts))
