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
        ctx.logger.info("MLIR parsing successful!")
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
        cdims = matcher.contraction_dimensions
        assert cdims, "no contraction dimensions"
        assert matcher.lhs_dims, "no lhs dimensions"
        assert matcher.rhs_dims, "no rhs dimensions"
        assert matcher.res_dims, "no result dimensions"
        assert len(cdims.batch) <= 1, f"must have at most 1 batch dimension"
        assert len(cdims.m) == 1, f"must have a single m dimension"
        assert len(cdims.n) == 1, f"must have a single n dimension"
        assert len(cdims.k) == 1, f"must have a single k dimension"
        lhs_type = ir.RankedTensorType(contraction_op.operands[0].type)
        rhs_type = ir.RankedTensorType(contraction_op.operands[1].type)
        res_type = ir.RankedTensorType(contraction_op.operands[2].type)
        matmul_size = MatmulSize(
            lhs_type.shape[matcher.lhs_dims.index(cdims.m[0])],
            rhs_type.shape[matcher.rhs_dims.index(cdims.n[0])],
            lhs_type.shape[matcher.lhs_dims.index(cdims.k[0])],
        )
        if len(cdims.batch) == 1:
            matmul_size.B = lhs_type.shape[matcher.lhs_dims.index(cdims.batch[0])]
        return ProblemSize(
            matmul_size,
            lhs_type=ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=ShapedType(res_type.shape, res_type.element_type),
            dispatch_kind=DispatchKind.contraction,
        )


# TODO(Max191): Support more convolution types. Only NHWC convs are supported.
class ConvolutionOpInterfaceParser(DispatchParser):
    def __init__(self):
        self.supported_ops = ["linalg.conv_2d_nhwc_hwcf"]

    def supports(self, op_name: str) -> bool:
        for supported_op_name in self.supported_ops:
            if supported_op_name.split(".")[-1] in op_name:
                return True
        return False

    def get_conv_operation(
        self,
        ir_module: ir.Module,
    ) -> Optional[ir.Operation]:
        return match_root_op(ir_module, NamedOpMatcher(self.supported_ops))

    # TODO(Max191): Pass the ir_module directly instead of the template str.
    def get_shapes(self, template: list[str]) -> ProblemSize:
        ir_module = ir.Module.parse("\n".join(template))
        conv_op = match_root_op(ir_module, NamedOpMatcher(self.supported_ops))
        assert conv_op is not None, f"convolution op not found"
        lhs_type = ir.RankedTensorType(conv_op.operands[0].type)
        rhs_type = ir.RankedTensorType(conv_op.operands[1].type)
        res_type = ir.RankedTensorType(conv_op.operands[2].type)
        dim_info = ConvDimInfo.from_rhs_res(rhs_type, res_type)
        return ProblemSize(
            MatmulSize(
                M=dim_info.oh * dim_info.ow,
                N=dim_info.oc,
                K=dim_info.fh * dim_info.fw * dim_info.ic,
                B=dim_info.n,
            ),
            lhs_type=ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=ShapedType(res_type.shape, res_type.element_type),
            dispatch_kind=DispatchKind.conv,
        )
