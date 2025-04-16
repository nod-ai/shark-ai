# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest dispatch_parser_test.py
"""

import pytest

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
from iree.compiler.dialects import linalg  # type: ignore

from . import common
from . import dispatch_parser

from .test_utils import tuner_ctx


CONTRACTION_TEMPLATE = r"""
builtin.module{{
    func.func @test(%arg0: {lhs_type}, %arg1: {rhs_type}) -> {res_type} {{
        %cst = arith.constant 0.000000e+00 : f32
        %0 = tensor.empty() : {res_type}
        %1 = linalg.fill ins(%cst : f32) outs(%0 : {res_type}) -> {res_type}
        %2 = linalg.generic {{
            indexing_maps = [
                {lhs_map},
                {rhs_map},
                {res_map}],
            iterator_types = {iterator_types}}}
            {{root_op}}
            ins(%arg0, %arg1 : {lhs_type}, {rhs_type})
            outs(%1 : {res_type}) {{
        ^bb0(%in: f16, %in_0: f16, %out: f32):
            %3 = arith.extf %in : f16 to f32
            %4 = arith.extf %in_0 : f16 to f32
            %5 = arith.mulf %3, %4 : f32
            %6 = arith.addf %out, %5 : f32
            linalg.yield %6 : f32
        }} -> {res_type}
        return %2 : {res_type}
    }}
}}"""


def test_get_contraction_operation(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx

    with ir.Location.unknown():
        transpose_b_str = CONTRACTION_TEMPLATE.format(
            lhs_type=ir.RankedTensorType.get([16, 64], ir.F16Type.get()),
            rhs_type=ir.RankedTensorType.get([32, 64], ir.F16Type.get()),
            res_type=ir.RankedTensorType.get([16, 32], ir.F32Type.get()),
            lhs_map="affine_map<(d0, d1, d2) -> (d0, d2)>",
            rhs_map="affine_map<(d0, d1, d2) -> (d1, d2)>",
            res_map="affine_map<(d0, d1, d2) -> (d0, d1)>",
            iterator_types='["parallel", "parallel", "reduction"]',
        )
    module = ir.Module.parse(transpose_b_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op)
    shapes: common.ProblemSize = parser.get_problem_size()
    assert shapes.matmul_size.B == []
    assert shapes.matmul_size.M == [16]
    assert shapes.matmul_size.N == [32]
    assert shapes.matmul_size.K == [64]
    assert shapes.lhs_type.shape == [16, 64]
    assert isinstance(shapes.lhs_type.element_type, ir.F16Type)
    assert shapes.rhs_type.shape == [32, 64]
    assert isinstance(shapes.rhs_type.element_type, ir.F16Type)
    assert shapes.res_type.shape == [16, 32]
    assert isinstance(shapes.res_type.element_type, ir.F32Type)

    with ir.Location.unknown():
        bmm_transposed_inputs_str = CONTRACTION_TEMPLATE.format(
            lhs_type=ir.RankedTensorType.get([5, 8, 128], ir.F16Type.get()),
            rhs_type=ir.RankedTensorType.get([128, 40, 5], ir.F16Type.get()),
            res_type=ir.RankedTensorType.get([5, 40, 8], ir.F32Type.get()),
            lhs_map="affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>",
            rhs_map="affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>",
            res_map="affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>",
            iterator_types='["parallel", "parallel", "parallel", "reduction"]',
        )
    module = ir.Module.parse(bmm_transposed_inputs_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op)
    shapes = parser.get_problem_size()
    assert shapes.matmul_size.B == [5]
    assert shapes.matmul_size.M == [8]
    assert shapes.matmul_size.N == [40]
    assert shapes.matmul_size.K == [128]

    with ir.Location.unknown():
        bmm_transposed_inputs_str = CONTRACTION_TEMPLATE.format(
            lhs_type=ir.RankedTensorType.get(
                [16, 8, 15, 16, 64, 256], ir.F16Type.get()
            ),
            rhs_type=ir.RankedTensorType.get(
                [16, 9, 15, 16, 128, 256], ir.F16Type.get()
            ),
            res_type=ir.RankedTensorType.get([16, 8, 9, 16, 64, 128], ir.F32Type.get()),
            lhs_map="affine_map<(b0, m0, n0, k0, b1, m1, n1, k1) -> (b0, m0, k0, b1, m1, k1)>",
            rhs_map="affine_map<(b0, m0, n0, k0, b1, m1, n1, k1) -> (b0, n0, k0, b1, n1, k1)>",
            res_map="affine_map<(b0, m0, n0, k0, b1, m1, n1, k1) -> (b0, m0, n0, b1, m1, n1)>",
            iterator_types='["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "parallel", "reduction"]',
        )
    module = ir.Module.parse(bmm_transposed_inputs_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op)
    shapes = parser.get_problem_size()
    assert shapes.matmul_size.B == [16, 16]
    assert shapes.matmul_size.M == [8, 64]
    assert shapes.matmul_size.N == [9, 128]
    assert shapes.matmul_size.K == [15, 256]


def test_get_conv_operation(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<2x34x34x16xi8>, %arg1: tensor<3x3x16x16xi8>) -> tensor<2x32x32x16xi32> {
                %cst = arith.constant 0 : i32
                %0 = tensor.empty() : tensor<2x32x32x16xi32>
                %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x32x32x16xi32>) -> tensor<2x32x32x16xi32>
                %2 = linalg.conv_2d_nhwc_hwcf {root_op}
                    ins(%arg0, %arg1 : tensor<2x34x34x16xi8>, tensor<3x3x16x16xi8>)
                    outs(%1 : tensor<2x32x32x16xi32>) -> tensor<2x32x32x16xi32>
                return %2 : tensor<2x32x32x16xi32>
            }
        }"""
    module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op)
    assert (
        parser.has_valid_root_op()
    ), f"ConvolutionOpInterfaceParser does not support the op: {root_op.name}"


def test_get_mmt_tile_sizes(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[128, 320, 0],
        reduction=[0, 0, 32],
        subgroup_m_count=1,
        subgroup_n_count=4,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 0)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [], 0, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    lowering_config = compilation_info.lowering_config
    assert lowering_config.workgroup_tile_sizes == [128, 320, 0]
    assert lowering_config.reduction_tile_sizes == [0, 0, 32]


def test_get_conv_tile_sizes(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 1, 464, 320, 1, 1, 0],
        reduction=[0, 0, 0, 0, 0, 0, 16],
        subgroup_m_count=1,
        subgroup_n_count=4,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 1)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [256, 1, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    assert compilation_info.lowering_config.workgroup_tile_sizes == [
        1,
        1,
        464,
        320,
        1,
        1,
        0,
    ]
    assert compilation_info.lowering_config.reduction_tile_sizes == [
        0,
        0,
        0,
        0,
        0,
        0,
        16,
    ]


def test_parse_mlir(tuner_ctx: common.TunerContext) -> None:
    mlir_str = r"""
    builtin.module  {
    func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
    }
    }
"""
    mlir_module = dispatch_parser.parse_mlir(mlir_str, tuner_ctx)
    assert mlir_module is not None
    assert isinstance(mlir_module, ir.Module)
    assert isinstance(mlir_module.body.operations[0], func.FuncOp)
