# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from dataclasses import astuple, dataclass, field
from enum import Enum
from types import TracebackType
from typing import Optional
from typing import Any

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_gpu  # type: ignore


class CommonTypes:
    def __init__(self, ctx: ir.Context):
        assert ctx
        self.i1 = ir.IntegerType.get_signless(1, ctx)
        self.i8 = ir.IntegerType.get_signless(8, ctx)
        self.i16 = ir.IntegerType.get_signless(16, ctx)
        self.i32 = ir.IntegerType.get_signless(32, ctx)
        self.i64 = ir.IntegerType.get_signless(64, ctx)

        self.f8E4M3FNUZ = ir.Float8E4M3FNUZType.get(ctx)
        self.f8E5M2FNUZ = ir.Float8E5M2FNUZType.get(ctx)
        self.f16 = ir.F16Type.get(ctx)
        self.f32 = ir.F32Type.get(ctx)

        self.bf16 = ir.BF16Type.get(ctx)

    def getI64(self, value: int) -> ir.IntegerAttr:
        return ir.IntegerAttr.get(self.i64, value)


class TunerContext:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.mlir_ctx: ir.Context = ir.Context()
        self.logger: logging.Logger = logger or logging.getLogger("tune")
        self.type: CommonTypes = CommonTypes(self.mlir_ctx)

    def __enter__(self) -> "TunerContext":
        self.mlir_ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self.mlir_ctx.__exit__(exc_type, exc_value, traceback)


class DispatchKind(Enum):
    conv = 0
    contraction = 1


@dataclass
class ShapedType:
    shape: list[int]
    element_type: ir.IntegerType | ir.FloatType

    def rank(self) -> int:
        return len(self.shape)

    @property
    def bitwidth(self) -> int:
        return self.element_type.width

    def __str__(self) -> str:
        dim_to_str = lambda dim: str(dim) if dim != -1 else "?"
        return "x".join(map(dim_to_str, self.shape)) + "x" + str(self.element_type)


@dataclass
class ContractionSizes:
    """
    Represents the size of the iteration space along each contraction dimension.
    For example, the following is a simple batch mmt:
      linalg.generic ... indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, n, k)>,
          affine_map<(b, m, n, k) -> (b, m, n)>,
        ] ...
        ins(%lhs: tensor<4x8x32xf16>, %rhs: tensor<4x16x32xf16>)
        outs(%acc: tensor<4x8x16xf16>)
    The ContractionSizes would be:
      M = [8]
      N = [16]
      K = [32]
      B = [4]
    """

    M: list[int]
    N: list[int]
    K: list[int]
    B: list[int] = field(default_factory=list)


@dataclass
class ContractionDimensions:
    """
    Stores which dimensions of the iteration space belong to M, N, K, or Batch.
    For example, the following is a simple batch mmt:
    linalg.generic ... indexing_maps = [
        affine_map<(b, m, n, k) -> (b, m, k)>,
        affine_map<(b, m, n, k) -> (b, n, k)>,
        affine_map<(b, m, n, k) -> (b, m, n)>,
        ]
    The ContractionDimensions would be:
    M = [1]
    N = [2]
    K = [3]
    B = [0]
    """

    m: list[int]
    n: list[int]
    k: list[int]
    batch: list[int] = field(default_factory=list)


@dataclass
class ConvolutionDimensions:
    """
    Stores which dimensions of the iteration space belong to the convolution.
    For example, the following is a simple nhwc_fhwc conv:
    linalg.generic ... indexing_maps = [
        affine_map<(b, oh, ow, oc, fh, fw, ic) -> (b, oh + fh, ow + fw, ic)>,
        affine_map<(b, oh, ow, oc, fh, fw, ic) -> (oc, fh, fw, ic)>,
        affine_map<(b, oh, ow, oc, fh, fw, ic) -> (b, oh, ow, oc)>,
        ]
    The ConvolutionDimensions would be:
    batch = [0]
    outputImage = [1, 2]
    outputChannel = [3]
    filterLoop = [4, 5]
    inputChannel = [6]
    depth = []
    strides = [1, 1]
    dilations = [1, 1]
    """

    batch: list[int] = field(default_factory=list)
    outputImage: list[int] = field(default_factory=list)
    outputChannel: list[int] = field(default_factory=list)
    filterLoop: list[int] = field(default_factory=list)
    inputChannel: list[int] = field(default_factory=list)
    depth: list[int] = field(default_factory=list)
    strides: list[int] = field(default_factory=list)
    dilations: list[int] = field(default_factory=list)


@dataclass
class ProblemSize:
    """
    Represents a problem size for a contraction or convolution operation. When it is
    a convolution all fields including: contraction_dims, lhs_expr_dims, rhs_expr_dims,
    res_expr_dims and conv_dims are required to be set.

    For example, the following is a simple convolution:
    %conv = linalg.conv_2d_nhwc_hwcf
        {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
        ins(%6, %7 : tensor<2x62x66x960xf16>, tensor<3x3x960x640xf16>)
        outs(%13 : tensor<2x60x64x640xf32>) -> tensor<2x60x64x640xf32>

    The ProblemSize would be:
    matmul_size = ContractionSizes(M=[2, 60, 64], N=[640], K=[3, 3, 960], B=[]),
    lhs_type = ShapedType(shape=[2, 62, 66, 960], element_type=F16Type(f16)),
    rhs_type = ShapedType(shape=[3, 3, 960, 640], element_type=F16Type(f16)),
    res_type = ShapedType(shape=[2, 60, 64, 640], element_type=F32Type(f32)),
    dispatch_kind = DispatchKind.conv

    # The contraction_dims field is intuitive for gemms. For convolutions, it recorded the convolution
    # dimension to contraction dimension mapping. In igemm setting:
    # - [d0, d1, d2] map to [b, oh, ow], or m dimension of the gemm
    # - [d3] map to [oc], or n dimension of the gemm
    # - [d4, d5, d6] map to [fh, fw, ic], or k dimension of the gemm
    contraction_dims = ContractionDimensions(m=[0, 1, 2], n=[3], k=[4, 5, 6], batch=[]),

    # *expr_dims fields represent the dimensions that appear in the affine map result expressions at
    # each dimension of the operator tensor.
    lhs_expr_dims = [[0], [1, 4], [2, 5], [6]],
    rhs_expr_dims = [[4], [5], [6], [3]],
    res_expr_dims = [[0], [1], [2], [3]],
    conv_dims = ConvolutionDimensions(...)

    """

    matmul_size: ContractionSizes
    lhs_type: ShapedType
    rhs_type: ShapedType
    res_type: ShapedType
    dispatch_kind: DispatchKind
    contraction_dims: ContractionDimensions
    lhs_expr_dims: Optional[list[list[int]]] = None
    rhs_expr_dims: Optional[list[list[int]]] = None
    res_expr_dims: Optional[list[list[int]]] = None
    conv_dims: Optional[ConvolutionDimensions] = None

    @property
    def MNK(self) -> tuple[list[int], list[int], list[int]]:
        return (self.matmul_size.M, self.matmul_size.N, self.matmul_size.K)


def get_compatible_mfma_intrinsics(
    problem_size: ProblemSize,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic],
) -> list[iree_gpu.MMAIntrinsic]:
    def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
        mma_attr = iree_gpu.MMAIntrinsicAttr.get(mma_intrinsic).mma
        a_type, b_type, c_type = mma_attr.abc_element_types
        if not isinstance(problem_size.res_type.element_type, type(c_type)):
            return False
        if not isinstance(
            problem_size.lhs_type.element_type, type(a_type)
        ) or not isinstance(problem_size.rhs_type.element_type, type(b_type)):
            return False
        return True

    return list(filter(is_comptible, mma_intrinsics))


# The key name for GPUPipelineOptionsAttr in the translation info config dictionary.
GPU_PIPELINE_OPTIONS_KEY = "gpu_pipeline_options"
# The key name for llvm_func_attrs attribute in the translation info config dictionary.
LLVM_FUNC_ATTRS_KEY = "llvm_func_attrs"
# The Key name for the 'amdgpu-waves-per-eu' within the llvm_func_attrs attribute.
WAVES_PER_EU_KEY = "amdgpu-waves-per-eu"


def get_lowering_config(
    tuner_ctx: TunerContext,
    **kwargs: Any,
) -> iree_gpu.LoweringConfigAttr:
    lowering_config_dict: dict[str, Any] = {}
    for key, value in kwargs.items():
        # A local variable to hold the transformed value.
        promoted_value = value
        match key:
            case "workgroup" | "reduction" | "subgroup" | "promote_operands":
                if isinstance(value, list):
                    promoted_value = ir.ArrayAttr.get(
                        [tuner_ctx.type.getI64(x) for x in value]
                    )
                elif not isinstance(value, ir.ArrayAttr):
                    assert (
                        False
                    ), f"Unsupported type for key '{key}': {type(value).__name__}"
            case "subgroup_m_count" | "subgroup_n_count":
                if isinstance(value, int):
                    promoted_value = tuner_ctx.type.getI64(value)
                elif not isinstance(value, tuner_ctx.type.i64):
                    assert (
                        False
                    ), f"Unsupported type for key '{key}': {type(value).__name__}"
            case "mma_kind":
                if not isinstance(value, iree_gpu.MMAAttr):
                    assert (
                        False
                    ), f"Unsupported type for key '{key}': {type(value).__name__}"
            case _:
                assert False, f"Unhandled key in lowering configuration: {key}"

        lowering_config_dict[key] = promoted_value
    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    return iree_gpu.LoweringConfigAttr.get(lowering_config_attrs)


# Generate a config dictionary used in translation_info attribute.
def get_translation_info_config(
    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int
) -> ir.DictAttr:
    """
    Example IR
    translation_info = #iree_codegen.translation_info<
                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
                     llvm_func_attrs = {"amdgpu-waves-per-eu" = "3"}
                    }
                >
    """
    waves_per_eu_str = str(waves_per_eu)

    # Create the waves_per_eu dictionary attribute.
    waves_per_eu_dict = ir.DictAttr.get(
        {WAVES_PER_EU_KEY: ir.StringAttr.get(waves_per_eu_str)}
    )

    config_dict = ir.DictAttr.get(
        {
            GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
            LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
        }
    )

    return config_dict


def read_input_mlir(filename: str) -> list[str]:
    with open(filename, "r") as f:
        return f.readlines()


@dataclass
class ConvDimInfo:
    n: int
    oh: int
    ow: int
    oc: int
    fh: int
    fw: int
    ic: int

    @staticmethod
    def from_rhs_res(
        rhs_shaped_type: ShapedType,
        rhs_dims: Optional[list[list[int]]],
        res_shaped_type: ShapedType,
        res_dims: Optional[list[list[int]]],
        conv_dims: Optional[ConvolutionDimensions],
    ) -> "ConvDimInfo":
        assert rhs_dims is not None, "no rhs dimensions"
        assert res_dims is not None, "no result dimensions"
        assert conv_dims is not None, "no convolution dimensions"
        assert len(conv_dims.filterLoop) == 2, "only 2 filter loops supported"
        assert len(conv_dims.outputChannel) == 1, "only 1 output channel supported"
        assert len(conv_dims.inputChannel) == 1, "only 1 input channel supported"
        assert len(conv_dims.batch) == 1, "only 1 batch dimension supported"
        assert (
            len(conv_dims.outputImage) == 2
        ), "only 2 output image dimensions supported"

        rhs_shape = rhs_shaped_type.shape
        filter_dict = {
            inner_list[0]: index for index, inner_list in enumerate(rhs_dims)
        }
        fh = rhs_shape[filter_dict.get(conv_dims.filterLoop[0], -1)]
        fw = rhs_shape[filter_dict.get(conv_dims.filterLoop[1], -1)]
        oc = rhs_shape[filter_dict.get(conv_dims.outputChannel[0], -1)]
        ic = rhs_shape[filter_dict.get(conv_dims.inputChannel[0], -1)]

        res_shape = res_shaped_type.shape
        res_dict = {inner_list[0]: index for index, inner_list in enumerate(res_dims)}
        n = res_shape[res_dict.get(conv_dims.batch[0], -1)]
        oh = res_shape[res_dict.get(conv_dims.outputImage[0], -1)]
        ow = res_shape[res_dict.get(conv_dims.outputImage[1], -1)]

        return ConvDimInfo(n, oh, ow, oc, fh, fw, ic)

    @staticmethod
    def from_problem_size(problem_size: ProblemSize):
        return ConvDimInfo.from_rhs_res(
            problem_size.rhs_type,
            problem_size.rhs_expr_dims,
            problem_size.res_type,
            problem_size.res_expr_dims,
            problem_size.conv_dims,
        )


@dataclass
class MLIRTransformation:
    """Transformation of MLIR context"""

    template: list[str]
    modified: str
    embeddable: str
