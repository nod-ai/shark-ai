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
from iree.compiler.dialects import func, iree_codegen, linalg  # type: ignore

from . import common


def get_parent_function_name(root_op: ir.Operation) -> str:
    """
    Returns the parent function's symbol name from a root operation.
    """
    # FIXME: This assumes the immediate parent is a function, but the root op
    # could be nested inside other operations (e.g., scf.if).
    func_op = root_op.parent.opview
    assert isinstance(func_op, func.FuncOp), f"Expected func.func, got {func_op.name}"
    return ir.StringAttr(func_op.name).value


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


@dataclass
class ContractionOpInfo(OpInfo):
    dims: common.ContractionDimensions
    matmul_size: common.ContractionSizes
    lhs_type: common.ShapedType
    rhs_type: common.ShapedType
    res_type: common.ShapedType


@dataclass
class ConvolutionOpInfo(OpInfo):
    dims: common.ContractionDimensions
    matmul_size: common.ContractionSizes
    lhs_type: common.ShapedType
    rhs_type: common.ShapedType
    res_type: common.ShapedType

    batch_sizes: list[int]
    output_image_sizes: list[int]
    output_channel_sizes: list[int]
    filter_loop_sizes: list[int]
    input_channel_sizes: list[int]
    depth_sizes: list[int]
    strides: list[int]
    dilations: list[int]


class DispatchParser(metaclass=ABCMeta):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        self._root_op = root_op
        self._tuner_ctx = tuner_ctx
        self._op_info: Optional[OpInfo] = None

    def get_root_op(self) -> ir.Operation:
        return self._root_op

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
        root_op = self.get_root_op()
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        assert convolution_dims, "no convolution dimensions"

        batch_indices = list(convolution_dims.batch)
        output_image_indices = list(convolution_dims.output_image)
        output_channel_indices = list(convolution_dims.output_channel)
        filter_loop_indices = list(convolution_dims.filter_loop)
        input_channel_indices = list(convolution_dims.input_channel)
        depth_indices = list(convolution_dims.depth)
        strides = list(convolution_dims.strides)
        dilations = list(convolution_dims.dilations)

        res_maps = linalg.get_indexing_maps(root_op)
        indexing_maps = [map_attr.value for map_attr in res_maps]

        contraction_dims = common.ContractionDimensions(
            batch=depth_indices,
            m=batch_indices + output_image_indices,
            n=output_channel_indices,
            k=filter_loop_indices + input_channel_indices,
        )

        batch_sizes = (
            [self.get_iter_dim_size(d, 2, indexing_maps) for d in batch_indices]
            if batch_indices
            else []
        )
        output_image_sizes = (
            [self.get_iter_dim_size(d, 2, indexing_maps) for d in output_image_indices]
            if output_image_indices
            else []
        )
        output_channel_sizes = (
            [
                self.get_iter_dim_size(d, 2, indexing_maps)
                for d in output_channel_indices
            ]
            if output_channel_indices
            else []
        )
        filter_loop_sizes = (
            [self.get_iter_dim_size(d, 1, indexing_maps) for d in filter_loop_indices]
            if filter_loop_indices
            else []
        )
        input_channel_sizes = (
            [self.get_iter_dim_size(d, 0, indexing_maps) for d in input_channel_indices]
            if input_channel_indices
            else []
        )
        depth_sizes = (
            [self.get_iter_dim_size(d, 2, indexing_maps) for d in depth_indices]
            if depth_indices
            else []
        )

        matmul_size = common.ContractionSizes(
            B=depth_sizes,
            M=batch_sizes + output_image_sizes,
            N=output_channel_sizes,
            K=filter_loop_sizes + input_channel_sizes,
        )

        lhs_type = root_op.operands[0].type
        rhs_type = root_op.operands[1].type
        res_type = root_op.operands[2].type

        self._op_info: ConvolutionOpInfo = ConvolutionOpInfo(
            root_op=root_op,
            indexing_maps=indexing_maps,
            dims=contraction_dims,
            matmul_size=matmul_size,
            lhs_type=common.ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=common.ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=common.ShapedType(res_type.shape, res_type.element_type),
            batch_sizes=batch_sizes,
            output_image_sizes=output_image_sizes,
            output_channel_sizes=output_channel_sizes,
            filter_loop_sizes=filter_loop_sizes,
            input_channel_sizes=input_channel_sizes,
            depth_sizes=depth_sizes,
            strides=strides,
            dilations=dilations,
        )

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

    def get_op_info(self) -> ConvolutionOpInfo:
        return self._op_info


class AttentionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return iree_codegen.isa_attention_op(root_op)

    def get_op_info(self) -> OpInfo:
        # TODO: Implement AttentionOpInfo extraction.
        raise NotImplementedError("AttentionOpInfo not yet implemented")
