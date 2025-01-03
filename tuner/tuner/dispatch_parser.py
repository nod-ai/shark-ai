# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import math
import re
from abc import ABCMeta, abstractmethod

from .op_matchers import *
from .common import *


def parse_tensor_type(tensor_type: str) -> ShapedType:
    shaped_ty = ir.RankedTensorType(ir.Type.parse(tensor_type))
    assert shaped_ty
    return ShapedType(shaped_ty.shape, shaped_ty.element_type)


def get_contract_workgroup_sizes(
    configuration: iree_codegen.CompilationInfoAttr, tile_dims: str
) -> list[int]:
    m, n, _k = configuration.lowering_config.workgroup_tile_sizes

    workgroup_size = [1] * len(tile_dims)
    for idx, dim in enumerate(tile_dims):
        if dim == "m":
            workgroup_size[idx] = m
        if dim == "n":
            workgroup_size[idx] = n
        if dim == "k":
            workgroup_size[idx] = 0

    return workgroup_size


def get_contract_reduction_sizes(
    configuration: iree_codegen.CompilationInfoAttr, tile_dims: str
) -> list[int]:
    _m, _n, k = configuration.lowering_config.reduction_tile_sizes
    reduction_size = [0] * len(tile_dims)
    for idx, dim in enumerate(tile_dims):
        if dim == "k":
            reduction_size[idx] = k

    return reduction_size


class MlirRegex(Enum):
    ssa_value = r"%[a-zA-Z0-9-_]+"
    tensor_type = r"tensor<([^>]+)>"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def dps_ins_two_args() -> str:
        return rf"ins\({MlirRegex.ssa_value}, {MlirRegex.ssa_value} : (?P<LHS>{MlirRegex.tensor_type}), (?P<RHS>{MlirRegex.tensor_type})\)"

    @staticmethod
    def dps_outs_one_arg() -> str:
        return rf"outs\({MlirRegex.ssa_value} : (?P<RES>{MlirRegex.tensor_type})\)"


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


class MmtParser(DispatchParser):
    def supports(self, op_name: str) -> bool:
        return "matmul_transpose_b" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        mmt_re = None
        dps = None
        for line in template:
            if "linalg.generic" not in line:
                continue
            if r'iterator_types = ["parallel", "parallel", "reduction"]' not in line:
                continue
            # ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>)
            mmt_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            dps = re.search(mmt_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 2
            lhs_M, lhs_K = lhs_shaped_type.shape

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 2
            rhs_N, rhs_K = rhs_shaped_type.shape

            assert lhs_shaped_type.element_type == rhs_shaped_type.element_type
            assert lhs_K == rhs_K

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 2
            res_M, res_N = res_shaped_type.shape

            assert lhs_M == res_M
            assert rhs_N == res_N

            matmul_size = MatmulSize(
                lhs_shaped_type.shape[0],
                rhs_shaped_type.shape[0],
                lhs_shaped_type.shape[1],
            )
            return ProblemSize(
                matmul_size,
                lhs_type=lhs_shaped_type,
                rhs_type=rhs_shaped_type,
                res_type=res_shaped_type,
                dispatch_kind=DispatchKind.mmt,
            )
        assert mmt_re
        assert False, f"'{mmt_re}' not found in given context"


class ConvParser(DispatchParser):
    def supports(self, op_name: str) -> bool:
        return "conv_2d_nhwc_hwcf" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        for line in template:
            if "linalg.conv_2d_nhwc_hwcf" not in line:
                continue

            # ins(%19, %20 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs (%27 : tensor<2x32x32x1280xf32>)
            conv_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(conv_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 4

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 4

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 4

            dim_info = ConvDimInfo.from_rhs_res(rhs_shaped_type, res_shaped_type)
            return ProblemSize(
                MatmulSize(
                    M=dim_info.oh * dim_info.ow,
                    N=dim_info.oc,
                    K=dim_info.fh * dim_info.fw * dim_info.ic,
                    B=dim_info.n,
                ),
                lhs_shaped_type,
                rhs_shaped_type,
                res_shaped_type,
                DispatchKind.conv,
            )

        assert False, "Shape not found"


class ContractionParser(DispatchParser):
    def __init__(self, lhs_dims: str, rhs_dims: str, tile_dims: str):
        self.lhs_dims = lhs_dims
        self.rhs_dims = rhs_dims
        self.tile_dims = tile_dims

    def supports(self, op_name: str) -> bool:
        return "matmul_like" in op_name

    def is_broadcast_rhs_mmt_op(self, line: str) -> bool:
        if "linalg.generic" not in line:
            return False
        if (
            r'iterator_types = ["parallel", "parallel", "parallel", "reduction"]'
            not in line
        ):
            return False
        if (
            r"indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"
            not in line
        ):
            return False
        return True

    def is_broadcast_rhs_mmt(self, template: list[str]) -> bool:
        return any(self.is_broadcast_rhs_mmt_op(line) for line in template)

    def get_shapes_broadcast_rhs_mmt(self, template: list[str]) -> ProblemSize:
        for line in template:
            if not self.is_broadcast_rhs_mmt_op(line):
                continue

            # ins(%11, %12 : tensor<2x1024x1280xi8>, tensor<10240x1280xi8>) outs(%19 : tensor<2x1024x10240xi32>)
            bmmt_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(bmmt_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 3

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 2

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 3

            B0, M0, K0 = lhs_shaped_type.shape
            N1, K1 = rhs_shaped_type.shape
            B2, M2, N2 = res_shaped_type.shape
            assert B0 == B2
            assert M0 == M2
            assert N1 == N2
            assert K0 == K1
            return ProblemSize(
                MatmulSize(M0, N1, K0, B0),
                lhs_shaped_type,
                rhs_shaped_type,
                res_shaped_type,
                DispatchKind.broadcast_rhs_mmt,
            )

        assert False, "Shape not found"

    def get_shapes(self, template: list[str]) -> ProblemSize:
        if self.is_broadcast_rhs_mmt(template):
            return self.get_shapes_broadcast_rhs_mmt(template)

        for line in template:
            if "linalg.generic" not in line:
                continue
            if "lowering_config =" not in line:
                continue
            if '"reduction"' not in line:
                continue

            # ins(%7, %8 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
            cont_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(cont_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == len(self.lhs_dims)

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == len(self.rhs_dims)

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() >= 2

            M = math.prod(
                val if dim == "m" else 1
                for dim, val in zip(self.lhs_dims, lhs_shaped_type.shape)
            )
            N = math.prod(
                val if dim == "n" else 1
                for dim, val in zip(self.rhs_dims, rhs_shaped_type.shape)
            )
            K0 = math.prod(
                val if dim == "k" else 1
                for dim, val in zip(self.lhs_dims, lhs_shaped_type.shape)
            )
            K1 = math.prod(
                val if dim == "k" else 1
                for dim, val in zip(self.rhs_dims, rhs_shaped_type.shape)
            )
            assert K0 == K1

            return ProblemSize(
                MatmulSize(M, N, K0),
                lhs_type=lhs_shaped_type,
                rhs_type=rhs_shaped_type,
                res_type=res_shaped_type,
                dispatch_kind=DispatchKind.contraction,
            )

        assert False, "Shape not found"


class BatchMmtParser(DispatchParser):
    def supports(self, op_name: str) -> bool:
        return "batch_matmul_transpose_b" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        for line in template:
            if "linalg.generic" not in line:
                continue
            if (
                r'iterator_types = ["parallel", "parallel", "parallel", "reduction"]'
                not in line
            ):
                continue
            # ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>)
            bmmt_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(bmmt_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 3

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 3

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 3

            B0, M0, K0 = lhs_shaped_type.shape
            B1, N1, K1 = rhs_shaped_type.shape
            B2, M2, N2 = res_shaped_type.shape
            assert B0 == B1
            assert B0 == B2
            assert M0 == M2
            assert N1 == N2
            assert K0 == K1
            return ProblemSize(
                MatmulSize(M0, N1, K0, B0),
                lhs_shaped_type,
                rhs_shaped_type,
                res_shaped_type,
                DispatchKind.batch_mmt,
            )

        assert False, "Shape not found"


class BatchMatmulParser(DispatchParser):
    def __init__(self, lhs_dims: str, rhs_dims: str, tile_dims: str):
        self.lhs_dims = lhs_dims
        self.rhs_dims = rhs_dims
        self.tile_dims = tile_dims

    def supports(self, op_name: str) -> bool:
        return "batch_matmul" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        for line in template:
            if "linalg.batch_matmul" not in line:
                continue
            # ins(%9, %10 : tensor<64x72x1280xf16>, tensor<64x1280x1280xf16>)
            # outs(%12 : tensor<64x72x1280xf32>)
            cont_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(cont_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == len(self.lhs_dims)

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == len(self.rhs_dims)

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == lhs_shaped_type.rank()

            LHS = lhs_shaped_type.shape
            RHS = rhs_shaped_type.shape
            RES = res_shaped_type.shape

            B = math.prod(
                val if dim == "b" else 1 for dim, val in zip(self.lhs_dims, LHS)
            )
            B0 = math.prod(
                val if dim == "b" else 1 for dim, val in zip(self.lhs_dims, RHS)
            )
            B1 = math.prod(
                val if dim == "b" else 1 for dim, val in zip(self.lhs_dims, RES)
            )
            M = math.prod(
                val if dim == "m" else 1 for dim, val in zip(self.lhs_dims, LHS)
            )
            N = math.prod(
                val if dim == "n" else 1 for dim, val in zip(self.rhs_dims, RHS)
            )
            K0 = math.prod(
                val if dim == "k" else 1 for dim, val in zip(self.lhs_dims, LHS)
            )
            K1 = math.prod(
                val if dim == "k" else 1 for dim, val in zip(self.rhs_dims, RHS)
            )
            assert B == B0 and B == B1
            assert K0 == K1

            return ProblemSize(
                MatmulSize(M, N, K0, B),
                lhs_type=lhs_shaped_type,
                rhs_type=rhs_shaped_type,
                res_type=res_shaped_type,
                dispatch_kind=DispatchKind.batch_matmul,
            )

        assert False, "Shape not found"
