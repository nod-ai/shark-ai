# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import ClassVar, Type, cast, Optional, Callable, Dict
import inspect
import textwrap
import logging
from dataclasses import dataclass

import torch
from jinja2 import Environment, BaseLoader

from sharktank.kernels.base import *
from sharktank.types.tensors import dtype_to_serialized_short_name
from sharktank.utils.logging import get_logger
from iree.turbine.transforms.merger import Merger
from iree.turbine.support.ir_imports import Operation, MLIRError, IrType

logger = get_logger("sharktank.ops")
_JINJA2_ENVIRONMENT: Optional[Environment] = None


def _get_jinja2_env() -> Environment:
    global _JINJA2_ENVIRONMENT
    if _JINJA2_ENVIRONMENT is None:
        _JINJA2_ENVIRONMENT = Environment(loader=BaseLoader())
    return _JINJA2_ENVIRONMENT


@dataclass
class _Dim:
    dynamic: bool
    name: str


@dataclass
class _Dtype:
    name: str


class _StaticDimExpando:
    def __getattr__(self, n: str) -> "_Dim":
        return _Dim(False, n)


class _DynDimExpando:
    def __getattr__(self, n) -> "_Dim":
        return _Dim(True, n)


class _DtypeExpando:
    def __getattr__(self, n) -> "_Dtype":
        return _Dtype(n)


class MLIRTensor:
    shapes: ClassVar[tuple[_Dim, ...]]
    dtype: ClassVar[_Dtype]
    tensor: Optional[torch.Tensor]

    def __init__(self, t: Optional[torch.Tensor] = None):
        self.tensor = t

    def __class_getitem__(
        cls, shape_and_dtype: tuple[_Dim | _Dtype, ...]
    ) -> Type["MLIRTensor"]:
        """Syntax: `KernelBuffer[shape1, shape2, ..., shapeN, dtype]`"""

        if not isinstance(shape_and_dtype, tuple) or len(shape_and_dtype) < 2:
            raise TypeError(f"Expected at least 2 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        ty = shape_and_dtype[-1]

        if not all(isinstance(s, _Dim) for s in shape):
            raise TypeError(f"Expected shape to be a tuple of _Dim, got {shape}")
        if not isinstance(ty, _Dtype):
            raise TypeError(f"Expected dtype to be a _Dtype, got {ty}")

        shape = cast(tuple[_Dim, ...], shape)
        ty = cast(_Dtype, ty)

        class SubType(cls):
            shapes = shape
            dtype = ty

        return cast(Type["MLIRTensor"], SubType)


class MLIRSpec:
    mlir: str
    subs: dict

    def __init__(self, mlir: str, subs: dict = {}):
        self.mlir = mlir
        self.subs = subs


def mlir_kernel(
    *, inputs: tuple[type[MLIRTensor], ...], results: tuple[type[MLIRTensor], ...]
):
    """
    A decorator that allows a user to inject inline mlir kernels directly into
    the model.

    TODO:
        - Add user guide with examples.
        - Allow constant dimensions and dtypes for results, currently we only
          support result dimensions to be derived symbolically from inputs.
        - Add more input signature types (other than MLIRTensor) like MLIRInt,
          MLIRFloat, MLIRListOfTensor.
        - Add more default substitutions like passing the value of a resolved
          symbol as a substitution to jinja generator.
    """

    def fun(func: Callable[..., MLIRSpec]) -> Callable:
        sig = inspect.signature(func)
        params = sig.parameters
        args = list(params.keys())

        if len(args) != len(inputs) + len(results):
            raise TypeError(
                "Number of arguments to kernel should be same as the mlir_kernel spec"
            )

        input_args = args[: len(inputs)]
        result_args = args[len(inputs) :]

        # Create a dimension mapping to input dimensions

        @CustomOp.register(library=LIBRARY)
        class kernel(CustomOp):
            @property
            def signature(self) -> str:
                input_tensors = [f"Tensor {arg}" for arg in input_args]
                result_tensors = ["Tensor" for _ in result_args]
                return f'{func.__name__}({", ".join(input_tensors)}) -> ({", ".join(result_tensors)})'

            def select(self, sel: KernelSelection):
                # Create input descriptions.
                input_descs = [sel.arg_tensor(i) for i in range(len(input_args))]

                # Specialize static dimensions.
                for sym_ty, desc in zip(inputs, input_descs):
                    static_dims = [
                        i for i, dim in enumerate(sym_ty.shapes) if not dim.dynamic
                    ]
                    desc.specialize_dims(*static_dims)

                # Resolve shape and dtype symbols.
                dims = {}
                dtypes = {}
                for sym_ty, ty in zip(inputs, input_descs):
                    # Resolve shape symbols.
                    for sym_dim, dim in zip(sym_ty.shapes, ty.t.shape):
                        if sym_dim.name in dims:
                            if not sym_dim.dynamic and dims[sym_dim.name] != dim:
                                raise ValueError("Mismatched dim error")
                        else:
                            dims[sym_dim.name] = dim
                    # Resolve dtype symbols.
                    if sym_ty.dtype.name in dtypes:
                        if dtypes[sym_ty.dtype.name] != ty.t.dtype:
                            raise ValueError("Mismatched dtype error")
                    else:
                        dtypes[sym_ty.dtype.name] = ty.t.dtype

                # Specialize static dimensions on return type.
                for sym_ty in results:
                    resolved_shape = [dims[dim.name] for dim in sym_ty.shapes]
                    resolved_dtype = dtypes[sym_ty.dtype.name]
                    desc = sel.return_new_tensor(
                        size=resolved_shape, dtype=resolved_dtype
                    )
                    static_dims = [
                        i for i, dim in enumerate(sym_ty.shapes) if not dim.dynamic
                    ]
                    desc.specialize_dims(*static_dims)

            def generate(self, ksel: KernelSelection, kb: KernelBuilder):
                # Create input descriptions and types.
                input_values = [kb.arg_value(i) for i in range(len(input_args))]
                input_types = [RankedTensorType(val.type) for val in input_values]

                # Resolve shape and dtype symbols.
                dims = {}
                dtypes = {}
                for sym_ty, ty in zip(inputs, input_types):
                    # Resolve shape symbols.
                    for sym_dim, dim in zip(sym_ty.shapes, ty.shape):
                        if sym_dim.dynamic:
                            # For dynamic dimensions, map the dim to None.
                            dim = None

                        if sym_dim.name in dims:
                            if dims[sym_dim.name] != dim:
                                raise ValueError("Mismatched dim error")
                        else:
                            dims[sym_dim.name] = dim
                    # Resolve dtype symbols.
                    if sym_ty.dtype.name in dtypes:
                        if dtypes[sym_ty.dtype.name] != ty.element_type:
                            raise ValueError("Mismatched dtype error")
                    else:
                        dtypes[sym_ty.dtype.name] = ty.element_type

                # Get the MLIR spec.
                mlir_spec = func(*input_values, *([None] * len(result_args)))

                # Insert type aliases to the mlir_spec.
                mlir = self._get_type_aliases(dims, dtypes) + mlir_spec.mlir

                # Generate kernel name.
                kernel_name = self._get_kernel_name(func.__name__, dims, dtypes)

                # Try to check if the symbol table already has a generated
                # kernel for this specialization.
                symbol_name = None
                try:
                    symbol_name = kb.symbol_table[kernel_name]
                except KeyError:
                    pass

                # If this kernel is not already generated, generate it using
                # the mlir spec.
                if symbol_name is None:
                    # Generate the MLIR spec using jinja.
                    asm = (
                        _get_jinja2_env()
                        .from_string(mlir)
                        .render({"kernel_name": kernel_name, **mlir_spec.subs})
                    )
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
                    op = module_op.operation

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Generated kernel IR %s:\n%s", kernel_name, str(op)
                        )
                    merger = Merger(
                        op, kb.module_body.owner, target_symbol_table=kb.symbol_table
                    )
                    merger.merge()

                    symbol_name = kb.symbol_table[kernel_name]

                kb.yield_results(*call_function(symbol_name, *kb.arg_bindings))

            def _get_type_aliases(
                self, dims: Dict[str, Optional[int]], dtypes: Dict[str, IrType]
            ) -> str:
                aliases = ""
                for arg, sym_ty in zip(input_args, inputs):
                    mlir_shapes = [
                        "?" if dims[dim.name] is None else str(dims[dim.name])
                        for dim in sym_ty.shapes
                    ]
                    dtype = dtypes[sym_ty.dtype.name]
                    aliases += f'!{arg} = tensor<{"x".join(mlir_shapes)}x{dtype}>\n'
                    aliases += f"!{arg}_dtype = {dtype}\n"
                for arg, sym_ty in zip(result_args, results):
                    mlir_shapes = [
                        "?" if dims[dim.name] is None else str(dims[dim.name])
                        for dim in sym_ty.shapes
                    ]
                    dtype = dtypes[sym_ty.dtype.name]
                    aliases += f'!{arg} = tensor<{"x".join(mlir_shapes)}x{dtype}>\n'
                    aliases += f"!{arg}_dtype = {dtype}\n"
                return aliases

            def _get_kernel_name(
                self,
                prefix: str,
                dims: Dict[str, Optional[int]],
                dtypes: Dict[str, IrType],
            ) -> str:
                kernel_name = prefix

                # Add input args as suffix.
                kernel_name += "_"
                input_names = []
                for sym_ty in inputs:
                    input_dims = []
                    for sym_dim in sym_ty.shapes:
                        input_dim = sym_dim.name
                        if not sym_dim.dynamic:
                            input_dim += f"_{dims[sym_dim.name]}"
                        input_dims.append(input_dim)
                    input_name = (
                        "_".join(input_dims) + "_" + str(dtypes[sym_ty.dtype.name])
                    )
                    input_names.append(input_name)
                kernel_name += "_".join(input_names)

                # Add result args as suffix.
                result_names = []
                kernel_name += "_"
                for sym_ty in results:
                    result_dims = []
                    for sym_dim in sym_ty.shapes:
                        result_dim = sym_dim.name
                        if not sym_dim.dynamic:
                            result_dim += f"_{dims[sym_dim.name]}"
                        result_dims.append(result_dim)
                    result_name = (
                        "_".join(result_dims) + "_" + str(dtypes[sym_ty.dtype.name])
                    )
                    result_names.append(result_name)
                kernel_name += "_".join(result_names)

                return kernel_name

        return kernel

    return fun


StaticDim = _StaticDimExpando()
DynDim = _DynDimExpando()
Dtype = _DtypeExpando()

__all__ = ["StaticDim", "DynDim", "Dtype", "MLIRTensor", "mlir_kernel", "MLIRSpec"]
