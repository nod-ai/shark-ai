# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABCMeta, abstractmethod

from .op_matchers import *
from .common import *


def parse_mlir(mlir_text: str, ctx: TunerContext) -> ir.Module:
    mlir_module = None
    try:
        mlir_module = ir.Module.parse(mlir_text, ctx.mlir_ctx)
        ctx.logger.debug("MLIR parsing successful!")
    except ir.MLIRError as e:
        ctx.logger.error(f"Error parsing MLIR: {e}")
        raise RuntimeError(f"Error parsing MLIR: {e}")

    return mlir_module


class DispatchParser(metaclass=ABCMeta):
    @abstractmethod
    def supports(self, op_name: str) -> bool:
        """Check if the tuner can handle the type of operation represented by the input string."""
        pass

    @abstractmethod
    def get_shapes(self, template: list[str]) -> ProblemSize:
        """Extract problem size of the operation."""
        pass


# TODO(Max191): Support linalg named op versions of contraction ops. The
# current matchers only work for linalg.generic ops.
class ContractionOpInterfaceParser(DispatchParser):
    def supports(self, op_name: str) -> bool:
        return (
            "matmul_like" in op_name
            or "batch_matmul" in op_name
            or "batch_matmul_transpose_b" in op_name
            or "matmul_transpose_b" in op_name
        )

    def get_contraction_operation(
        self,
        ir_module: ir.Module,
    ) -> Optional[ir.Operation]:
        return match_root_op(ir_module, ContractionOpInterfaceMatcher())

    # TODO(Max191): Pass the ir_module directly instead of the template str.
    def get_shapes(self, template: list[str]) -> ProblemSize:
        matcher = ContractionOpInterfaceMatcher()
        ir_module = ir.Module.parse("\n".join(template))
        contraction_op = match_root_op(ir_module, matcher)
        assert contraction_op is not None, f"contraction op not found"
        contraction_dims = matcher.contraction_dimensions
        assert contraction_dims, "no contraction dimensions"
        assert matcher.lhs_dims, "no lhs dimensions"
        assert matcher.rhs_dims, "no rhs dimensions"
        assert matcher.res_dims, "no result dimensions"
        lhs_type = ir.RankedTensorType(contraction_op.operands[0].type)
        rhs_type = ir.RankedTensorType(contraction_op.operands[1].type)
        res_type = ir.RankedTensorType(contraction_op.operands[2].type)
        matmul_size = ContractionSizes(
            M=[
                lhs_type.shape[matcher.lhs_dims.index(dim)]
                for dim in contraction_dims.m
            ],
            N=[
                rhs_type.shape[matcher.rhs_dims.index(dim)]
                for dim in contraction_dims.n
            ],
            K=[
                lhs_type.shape[matcher.lhs_dims.index(dim)]
                for dim in contraction_dims.k
            ],
            B=[
                lhs_type.shape[matcher.lhs_dims.index(dim)]
                for dim in contraction_dims.batch
            ],
        )
        return ProblemSize(
            matmul_size,
            lhs_type=ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=ShapedType(res_type.shape, res_type.element_type),
            dispatch_kind=DispatchKind.contraction,
            contraction_dims=contraction_dims,
            lhs_expr_dims=[[d] for d in matcher.lhs_dims],
            rhs_expr_dims=[[d] for d in matcher.rhs_dims],
            res_expr_dims=[[d] for d in matcher.res_dims],
        )


# TODO(Max191): Support more convolution types. Only NHWC convs are supported.
class ConvolutionOpInterfaceParser(DispatchParser):
    def __init__(self):
        self.supported_ops = ["conv"]

    def supports(self, op_name: str) -> bool:
        for supported_op_name in self.supported_ops:
            if supported_op_name.split(".")[-1] in op_name:
                return True
        return False

    def get_conv_operation(
        self,
        ir_module: ir.Module,
    ) -> Optional[ir.Operation]:
        return match_root_op(ir_module, ConvolutionOpInterfaceMatcher())

    # TODO(Max191): Pass the ir_module directly instead of the template str.
    def get_shapes(self, template: list[str]) -> ProblemSize:
        ir_module = ir.Module.parse("\n".join(template))
        matcher = ConvolutionOpInterfaceMatcher()
        conv_op = match_root_op(ir_module, matcher)
        conv_dims = matcher.convolution_dimensions

        assert conv_op is not None, f"convolution op not found"

        lhs_type = ir.RankedTensorType(conv_op.operands[0].type)
        rhs_type = ir.RankedTensorType(conv_op.operands[1].type)
        res_type = ir.RankedTensorType(conv_op.operands[2].type)
        dim_info = ConvDimInfo.from_rhs_res(rhs_type, res_type)
        return ProblemSize(
            matmul_size=ContractionSizes(
                M=[dim_info.n, dim_info.oh, dim_info.ow],
                N=[dim_info.oc],
                K=[dim_info.fh, dim_info.fw, dim_info.ic],
            ),
            lhs_type=ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=ShapedType(res_type.shape, res_type.element_type),
            dispatch_kind=DispatchKind.conv,
            contraction_dims=ContractionDimensions(
                m=[0, 1, 2],
                n=[3],
                k=[4, 5, 6],
            ),
            lhs_expr_dims=matcher.lhs_expr_dims,
            rhs_expr_dims=matcher.rhs_expr_dims,
            res_expr_dims=matcher.res_expr_dims,
            conv_dims=conv_dims,
        )
