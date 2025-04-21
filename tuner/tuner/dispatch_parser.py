# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABCMeta, abstractmethod

from iree.compiler.dialects import linalg  # type: ignore

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
    def __init__(self, root_op: ir.Operation):
        self._root_op = root_op

    def get_root_op(self) -> ir.Operation:
        return self._root_op

    @abstractmethod
    def has_valid_root_op(self) -> bool:
        """Check if the root_op is valid and supported by this tuner."""
        pass

    @abstractmethod
    def get_problem_size(self) -> ProblemSize:
        """Extract problem size of the operation."""
        pass


class ContractionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return linalg.isa_contraction_op(root_op)

    def get_problem_size(self) -> ProblemSize:
        root_op = self.get_root_op()
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        assert contraction_dims, "no contraction dimensions"

        res_maps = linalg.get_indexing_maps(root_op)
        maps = [map_attr.value for map_attr in res_maps]
        lhs_dims = get_map_result_dim_positions(maps[0])
        rhs_dims = get_map_result_dim_positions(maps[1])
        res_dims = get_map_result_dim_positions(maps[2])

        lhs_type = ir.RankedTensorType(root_op.operands[0].type)
        rhs_type = ir.RankedTensorType(root_op.operands[1].type)
        res_type = ir.RankedTensorType(root_op.operands[2].type)

        assert lhs_dims, "no lhs dimensions"
        assert rhs_dims, "no rhs dimensions"
        assert res_dims, "no result dimensions"

        matmul_size = ContractionSizes(
            M=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.m],
            N=[rhs_type.shape[rhs_dims.index(dim)] for dim in contraction_dims.n],
            K=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.k],
            B=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch],
        )
        return ProblemSize(
            matmul_size,
            lhs_type=ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=ShapedType(res_type.shape, res_type.element_type),
            dispatch_kind=DispatchKind.contraction,
            contraction_dims=contraction_dims,
        )


# TODO(Max191): Support more convolution types. Only NHWC convs are supported.
class ConvolutionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)
        self.supported_ops = ["linalg.conv_2d_nhwc_hwcf"]

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        if not linalg.isa_convolution_op(root_op):
            return False

        return root_op.name in self.supported_ops

    def get_problem_size(self) -> ProblemSize:
        root_op = self.get_root_op()
        lhs_type = ir.RankedTensorType(root_op.operands[0].type)
        rhs_type = ir.RankedTensorType(root_op.operands[1].type)
        res_type = ir.RankedTensorType(root_op.operands[2].type)
        dim_info = ConvDimInfo.from_rhs_res(rhs_type, res_type)
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        assert convolution_dims, "no convolution dimensions"
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
                m=list(convolution_dims.batch) + list(convolution_dims.output_image),
                n=list(convolution_dims.output_channel),
                k=list(convolution_dims.filter_loop)
                + list(convolution_dims.input_channel),
            ),
        )
