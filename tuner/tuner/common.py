# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import logging
from dataclasses import astuple, dataclass
from enum import Enum
from typing import Optional
from typing import Any

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_gpu  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore


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
    def __init__(self, mlir_ctx: ir.Context, logger: logging.Logger):
        self.mlir_ctx: ir.Context = mlir_ctx
        self.logger: logging.Logger = logger
        self.type: CommonTypes = CommonTypes(mlir_ctx)


class DispatchKind(Enum):
    conv = 1
    mmt = 2
    contraction = 3
    batch_mmt = 4
    batch_matmul = 5
    broadcast_rhs_mmt = 6


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
class MatmulSize:
    M: int
    N: int
    K: int
    B: int = 1


@dataclass
class ContractionSizes:
    M: list[int]
    N: list[int]
    K: list[int]
    B: list[int]


@dataclass
class ContractionDimensions:
    batch: list[int]
    m: list[int]
    n: list[int]
    k: list[int]


@dataclass
class ProblemSize:
    matmul_size: MatmulSize | ContractionSizes
    lhs_type: ShapedType
    rhs_type: ShapedType
    res_type: ShapedType
    dispatch_kind: DispatchKind
    cdims: Optional[ContractionDimensions] = None

    @property
    def MNK(self) -> tuple[int | list[int], int | list[int], int | list[int]]:
        return (self.matmul_size.M, self.matmul_size.N, self.matmul_size.K)

    @property
    def MNK_lists(self) -> tuple[list[int], list[int], list[int]]:
        if isinstance(self.matmul_size, MatmulSize):
            return ([self.matmul_size.M], [self.matmul_size.N], [self.matmul_size.K])
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
        if problem_size.dispatch_kind != DispatchKind.batch_matmul:
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
            case "workgroup" | "reduction" | "subgroup":
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
    def from_rhs_res(rhs_shaped_type: ShapedType, res_shaped_type: ShapedType):
        n, oh, ow, oc = res_shaped_type.shape
        fh, fw, ic, _ = rhs_shaped_type.shape
        return ConvDimInfo(n, oh, ow, oc, fh, fw, ic)

    @staticmethod
    def from_problem_size(problem_size: ProblemSize):
        return ConvDimInfo.from_rhs_res(problem_size.rhs_type, problem_size.res_type)


@dataclass
class MLIRTransformation:
    """Transformation of MLIR context"""

    template: list[str]
    modified: str
    embeddable: str
