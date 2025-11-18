# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence, Tuple

import logging
from pathlib import Path
import textwrap

from jinja2 import Environment, PackageLoader, select_autoescape

from iree.turbine.support.ir_imports import (
    FlatSymbolRefAttr,
    FunctionType,
    IrType,
    MLIRError,
    Operation,
    RankedTensorType,
    StringAttr,
    TypeAttr,
    Value,
)

from iree.turbine.runtime.op_reg import (
    def_library,
    CustomOp,
    KernelBuilder,
    KernelSelection,
    TensorArg,
)

from iree.turbine.transforms.merger import Merger

from amdsharktank.utils.logging import get_logger

LIBRARY = def_library("amdsharktank")
TEMPLATES_DIR = Path(__file__).parent / "templates"
logger = get_logger("amdsharktank.ops")

_JINJA2_ENVIRONMENT: Optional[Environment] = None


__all__ = [
    "call_function",
    "inline_template_function",
    "load_jinja_template",
    "unpack_tensor_type",
    "specialize_all_known_dims",
    "CustomOp",
    "KernelBuilder",
    "KernelSelection",
    "RankedTensorType",
    "LIBRARY",
]


################################################################################
# Templating helpers.
# Consider adding these upstream.
################################################################################


def unpack_tensor_type(tensor_type: IrType) -> Tuple[str, str, IrType]:
    """Unpacks a RankedTensorType into components usually needed for templating.

    Returns:
        * The stringified asm form.
        * An "identifier friendly" form of the shape and element type.
        * The raw element type.
    """
    rtt = RankedTensorType(tensor_type)
    ident = f"{'x'.join([str(dim) if dim >= 0 else 'D' for dim in rtt.shape])}x{rtt.element_type}"
    return str(rtt), ident, rtt.element_type


def specialize_all_known_dims(tensor_arg: TensorArg):
    """Specializes all dimensions of a tensor arg that are known.

    If a dimension is an `int`, it is specialized. Otherwise (i.e. SymInt) it
    is left dynamic.
    """
    spec_dims = tensor_arg.spec_dims
    for index, dim in enumerate(tensor_arg.t.shape):
        if isinstance(dim, int):
            spec_dims[index] = dim


################################################################################
# Jinja template manipulation.
# TODO: This is upstream now. Just use that.
################################################################################


def _get_jinja2_env() -> Environment:
    global _JINJA2_ENVIRONMENT
    if _JINJA2_ENVIRONMENT is None:
        _JINJA2_ENVIRONMENT = Environment(loader=PackageLoader(__name__, "templates"))
    return _JINJA2_ENVIRONMENT


def call_function(target_function: Operation, *operands: Value) -> Sequence[Value]:
    target_symbol = FlatSymbolRefAttr.get(target_function.attributes["sym_name"].value)
    ftype = FunctionType(TypeAttr(target_function.attributes["function_type"]).value)
    operands = [i for i in operands if i is not None]
    return Operation.create(
        "util.call",
        results=ftype.results,
        operands=operands,
        attributes={
            "callee": target_symbol,
        },
    ).results


def inline_template_function(
    kb: KernelBuilder,
    template_file: str,
    function_name: str,
    template_type: str = "format",
    **kwargs,
) -> Operation:
    """Inlines a template module by first expanding its ASM via **kwargs.

    Returns the inlined symbol `function_name`, which is expected to have been
    in the template.
    """
    try:
        return kb.symbol_table[function_name]
    except KeyError:
        pass
    source_module_op = load_jinja_template(kb, template_file, **kwargs)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated kernel IR %s:\n%s", function_name, str(source_module_op)
        )
    merger = Merger(
        source_module_op, kb.module_body.owner, target_symbol_table=kb.symbol_table
    )
    merger.merge()
    return kb.symbol_table[function_name]


def load_jinja_template(kb: KernelBuilder, template_file: str, **kwargs) -> Operation:
    """Loads an MLIR jinja-based template by name.

    The file is loaded relative to the templates/ directory. It is interpolated
    with **kwargs and loaded into the KernelBuilder.
    """
    asm = _get_jinja2_env().get_template(template_file).render(**kwargs)
    try:
        module_op = Operation.parse(asm, context=kb.context)
    except MLIRError as e:
        lines = asm.splitlines()
        lines_numbered = "\n".join(
            [f"      {str(i+1):>5}: {l}" for i, l in enumerate(lines)]
        )
        raise RuntimeError(
            f"Error parsing generated op template:"
            f"\n{textwrap.indent(str(e), '  ')}"
            f"\n{lines_numbered}"
        )
    return module_op.operation
