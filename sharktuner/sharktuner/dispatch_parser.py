# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class OpInfo:
    root_op: ir.Operation
    indexing_maps: list[ir.AffineMap]

    @property
    def parent_function_name(self) -> str:
        """
        Extracts the parent function name and prefixes it with "match_".
        """
        func_op = self.root_op.parent.opview
        assert isinstance(
            func_op, func.FuncOp
        ), f"Expected func.func, got {func_op.name}"
        func_name_attr = func_op.name
        return f"match_{ir.StringAttr(func_name_attr).value}"


@dataclass
class ContractionOpInfo(OpInfo):
    dims: common.ContractionDimensions
    matmul_size: common.ContractionSizes
    lhs_type: common.ShapedType
    rhs_type: common.ShapedType
    res_type: common.ShapedType


class DispatchParser(metaclass=ABCMeta):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        self._root_op = root_op
        self._tuner_ctx = tuner_ctx
        self._op_info: Optional[OpInfo] = None

    def get_root_op(self) -> ir.Operation:
        return self._root_op

    def get_parent_function_name(self) -> str:
        return (
            self._op_info.parent_function_name
            if self._op_info
            else self._compute_parent_function_name()
        )

    # TODO(Bangtian): This is a temporary solution for convolution and attention ops,
    # which don't yet implement get_op_info() and thus have _op_info set to None.
    # Once ConvolutionOpInfo and AttentionOpInfo are implemented, this
    # fallback method can be removed.
    def _compute_parent_function_name(self) -> str:
        """
        Helper to compute parent function name before op_info is created.
        """
        func_op = self._root_op.parent.opview
        assert isinstance(
            func_op, func.FuncOp
        ), f"Expected func.func, got {func_op.name}"
        func_name_attr = func_op.name
        return f"match_{ir.StringAttr(func_name_attr).value}"

    def get_iter_dim_size(
        self, iter_dim: int, operand_idx: int, indexing_maps: list[ir.AffineMap]
    ) -> int:
        root_op = self.get_root_op()
        operand_type = root_op.operands[operand_idx].type
        indexing_map = indexing_maps[operand_idx]
        tensor_dim = list(indexing_map.results).index(ir.AffineExpr.get_dim(iter_dim))
        return operand_type.shape[tensor_dim]

    @abstractmethod
    def has_valid_root_op(self) -> bool:
        """Check if the root_op is valid and supported by this tuner."""
        pass

    @abstractmethod
    def get_op_info(self) -> OpInfo:
        """Extract and return OpInfo for this operation."""
        pass


class ContractionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)
        root_op = self.get_root_op()
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        assert contraction_dims, "no contraction dimensions"
        dims = common.ContractionDimensions(
            batch=list(contraction_dims.batch),
            m=list(contraction_dims.m),
            n=list(contraction_dims.n),
            k=list(contraction_dims.k),
        )
        res_maps = linalg.get_indexing_maps(root_op)
        indexing_maps = [map_attr.value for map_attr in res_maps]

        lhs_dims = common.get_map_result_dim_positions(indexing_maps[0])
        rhs_dims = common.get_map_result_dim_positions(indexing_maps[1])
        res_dims = common.get_map_result_dim_positions(indexing_maps[2])

        assert lhs_dims, "no lhs dimensions"
        assert rhs_dims, "no rhs dimensions"
        assert res_dims, "no result dimensions"

        lhs_type = ir.RankedTensorType(root_op.operands[0].type)
        rhs_type = ir.RankedTensorType(root_op.operands[1].type)
        res_type = ir.RankedTensorType(root_op.operands[2].type)

        matmul_size = common.ContractionSizes(
            M=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.m],
            N=[rhs_type.shape[rhs_dims.index(dim)] for dim in contraction_dims.n],
            K=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.k],
            B=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch],
        )

        self._op_info: ContractionOpInfo = ContractionOpInfo(
            root_op=root_op,
            indexing_maps=indexing_maps,
            dims=dims,
            matmul_size=matmul_size,
            lhs_type=common.ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=common.ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=common.ShapedType(res_type.shape, res_type.element_type),
        )

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return linalg.isa_contraction_op(root_op)

    def get_op_info(self) -> ContractionOpInfo:
        return self._op_info


class ConvolutionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

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

    def get_op_info(self) -> OpInfo:
        # TODO: Implement ConvolutionOpInfo extraction.
        raise NotImplementedError("ConvolutionOpInfo not yet implemented")


class AttentionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return iree_codegen.isa_attention_op(root_op)

    def get_op_info(self) -> OpInfo:
        # TODO: Implement AttentionOpInfo extraction.
        raise NotImplementedError("AttentionOpInfo not yet implemented")
