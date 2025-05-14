# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABCMeta, abstractmethod

from iree.compiler.dialects import linalg, func  # type: ignore

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

    def get_problem_size(self) -> ProblemSize:
        root_op = self.get_root_op()

        def find_iter_dim_size(iter_dim: int, operand: int):
            operand_type = root_op.operands[operand].type
            indexing_map = linalg.get_indexing_maps(root_op)[operand]
            tensor_dim = list(indexing_map.value.results).index(
                ir.AffineExpr.get_dim(iter_dim)
            )
            return operand_type.shape[tensor_dim]

        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        assert convolution_dims, "no convolution dimensions"
        contraction_dims = ContractionDimensions(
            batch=list(convolution_dims.depth),
            m=list(convolution_dims.batch) + list(convolution_dims.output_image),
            n=list(convolution_dims.output_channel),
            k=list(convolution_dims.filter_loop) + list(convolution_dims.input_channel),
        )
        # Parallel dimension sizes come from the output operand, and reduction
        # dimension sizes come from filter.
        matmul_size = ContractionSizes(
            B=[find_iter_dim_size(d, operand=2) for d in contraction_dims.batch],
            M=[find_iter_dim_size(d, operand=2) for d in contraction_dims.m],
            N=[find_iter_dim_size(d, operand=2) for d in contraction_dims.n],
            K=[find_iter_dim_size(d, operand=1) for d in contraction_dims.k],
        )

        lhs_type = root_op.operands[0].type
        rhs_type = root_op.operands[1].type
        res_type = root_op.operands[2].type
        return ProblemSize(
            matmul_size=matmul_size,
            lhs_type=ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=ShapedType(res_type.shape, res_type.element_type),
            dispatch_kind=DispatchKind.conv,
            contraction_dims=contraction_dims,
        )
