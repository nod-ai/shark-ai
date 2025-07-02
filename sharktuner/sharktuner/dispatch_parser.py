# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABCMeta, abstractmethod

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import linalg, func  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore

from . import common


def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
    mlir_module = None
    try:
        mlir_module = ir.Module.parse(mlir_text, ctx.mlir_ctx)
        ctx.logger.debug("MLIR parsing successful!")
    except ir.MLIRError as e:
        ctx.logger.error(f"Error parsing MLIR: {e}")
        raise RuntimeError(f"Error parsing MLIR: {e}")

    return mlir_module


class DispatchParser(metaclass=ABCMeta):
    def __init__(self, root_op: ir.Operation):
        self._root_op = root_op
        func_op = self._root_op.parent.opview
        assert isinstance(
            func_op, func.FuncOp
        ), f"Expected func.func, got {func_op.name}"
        func_name_attr = func_op.name
        self._func_name = f"match_{ir.StringAttr(func_name_attr).value}"

    def get_root_op(self) -> ir.Operation:
        return self._root_op

    def get_root_op_func_name(self) -> str:
        return self._func_name

    @abstractmethod
    def has_valid_root_op(self) -> bool:
        """Check if the root_op is valid and supported by this tuner."""
        pass


class ContractionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return linalg.isa_contraction_op(root_op)


class ConvolutionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        if not linalg.isa_convolution_op(root_op):
            return False
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        assert convolution_dims, "no convolution dimensions"
        # Only allow 'nhwc_hwcf' convs.
        # TODO: This dispatch parser class supports more layouts, but constraint
        #       generation is not tested. Relax this check as support is verified.
        if (
            list(convolution_dims.batch) != [0]
            or list(convolution_dims.output_image) != [1, 2]
            or list(convolution_dims.output_channel) != [3]
            or list(convolution_dims.filter_loop) != [4, 5]
            or list(convolution_dims.input_channel) != [6]
            or list(convolution_dims.depth) != []
        ):
            return False
        return True


class AttentionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return iree_codegen.isa_attention_op(root_op)
