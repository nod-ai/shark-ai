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


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    from logging import Logger
    from unittest.mock import MagicMock

    with ir.Context() as ctx:
        logger: Logger = MagicMock(spec=Logger)
        yield common.TunerContext(ctx, logger)


def test_parse_tensor_type(tuner_ctx: common.TunerContext) -> None:
    assert dispatch_parser.parse_tensor_type("tensor<1x2x3xf32>") == common.ShapedType(
        [1, 2, 3], tuner_ctx.type.f32
    )
    assert dispatch_parser.parse_tensor_type("tensor<123xi8>") == common.ShapedType(
        [123], tuner_ctx.type.i8
    )


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
    parser = dispatch_parser.ContractionOpInterfaceParser()
    mmt_op = parser.get_contraction_operation(module)
    assert mmt_op is not None
    assert isinstance(mmt_op.opview, linalg.GenericOp)
    shapes: common.ProblemSize = parser.get_shapes(transpose_b_str.splitlines())
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
    mmt_op = parser.get_contraction_operation(module)
    shapes = parser.get_shapes(bmm_transposed_inputs_str.splitlines())
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
    mmt_op = parser.get_contraction_operation(module)
    shapes = parser.get_shapes(bmm_transposed_inputs_str.splitlines())
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
    parser = dispatch_parser.ConvolutionOpInterfaceParser()
    conv_op = parser.get_conv_operation(module)
    assert conv_op is not None
    assert isinstance(conv_op.opview, linalg.Conv2DNhwcHwcfOp)


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


def test_get_contract_tile_sizes(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[4, 8, 0],
        reduction=[0, 0, 16],
        subgroup_m_count=1,
        subgroup_n_count=1,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 32, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    assert dispatch_parser.get_contract_workgroup_sizes(compilation_info, "mnk") == [
        4,
        8,
        0,
    ]
    assert dispatch_parser.get_contract_reduction_sizes(compilation_info, "mnk") == [
        0,
        0,
        16,
    ]
    assert dispatch_parser.get_contract_workgroup_sizes(compilation_info, "nmk") == [
        8,
        4,
        0,
    ]
    assert dispatch_parser.get_contract_reduction_sizes(compilation_info, "nmk") == [
        0,
        0,
        16,
    ]
    assert dispatch_parser.get_contract_workgroup_sizes(compilation_info, "knm") == [
        0,
        8,
        4,
    ]
    assert dispatch_parser.get_contract_reduction_sizes(compilation_info, "knm") == [
        16,
        0,
        0,
    ]
    assert dispatch_parser.get_contract_workgroup_sizes(compilation_info, "kkk") == [
        0,
        0,
        0,
    ]
    assert dispatch_parser.get_contract_reduction_sizes(compilation_info, "kkk") == [
        16,
        16,
        16,
    ]


def test_get_shapes_mmt(tuner_ctx: common.TunerContext) -> None:
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert dispatch_parser.MmtParser().get_shapes(template) == common.ProblemSize(
        common.MatmulSize(2048, 1280, 1280),
        common.ShapedType([2048, 1280], tuner_ctx.type.f16),
        common.ShapedType([1280, 1280], tuner_ctx.type.f16),
        common.ShapedType([2048, 1280], tuner_ctx.type.f32),
        dispatch_parser.DispatchKind.mmt,
    )


def test_get_shapes_conv(tuner_ctx: common.TunerContext) -> None:
    template = [
        r"%7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>} ins(%cst : f32) outs(%4 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"%8 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 32, 256, 1, 1, 32]]>, strides = dense<1> : vector<2xi64>} ins(%5, %6 : tensor<1x3x34x1280xf16>, tensor<3x3x1280x256xf16>) outs(%7 : tensor<1x1x32x256xf32>) -> tensor<1x1x32x256xf32>",
        r"flow.dispatch.tensor.store %8, %2, offsets = [%workgroup_id_z, %workgroup_id_y, 0, %3], sizes = [1, 1, 32, 256], strides = [1, 1, 1, 1] : tensor<1x1x32x256xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>",
    ]
    assert dispatch_parser.ConvParser().get_shapes(template) == common.ProblemSize(
        common.MatmulSize(32, 256, 11520),
        common.ShapedType([1, 3, 34, 1280], tuner_ctx.type.f16),
        common.ShapedType([3, 3, 1280, 256], tuner_ctx.type.f16),
        common.ShapedType([1, 1, 32, 256], tuner_ctx.type.f32),
        dispatch_parser.DispatchKind.conv,
    )


def test_get_shapes_contract(tuner_ctx: common.TunerContext) -> None:
    template = [
        r"%18 = tensor.empty() : tensor<2048x1280xf32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} ins(%cst : f32) outs(%18 : tensor<2048x1280xf32>) -> tensor<2048x1280xf32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 128, 64]]>} {',
        r"^bb0(%in: f16, %in_0: f16, %out: f32):",
    ]
    assert dispatch_parser.ContractionParser("mk", "nk", "mnk").get_shapes(
        template
    ) == common.ProblemSize(
        common.MatmulSize(2048, 1280, 1280),
        common.ShapedType([2048, 1280], tuner_ctx.type.f16),
        common.ShapedType([1280, 1280], tuner_ctx.type.f16),
        common.ShapedType([2048, 1280], tuner_ctx.type.f32),
        dispatch_parser.DispatchKind.contraction,
    )


def test_get_shapes_batch_matmul(tuner_ctx: common.TunerContext) -> None:
    template = [
        "%10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "%11 = linalg.batch_matmul ins(%8, %9 : tensor<1x32x1024xf32>, tensor<1x1024x32xf32>) outs(%10 : tensor<1x32x32xf32>) -> tensor<1x32x32xf32>",
        "flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 32, 32], strides = [1, 1, 1] : tensor<1x32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x32x64xf32>>",
    ]
    assert dispatch_parser.BatchMatmulParser("bmk", "bkn", "mnk").get_shapes(
        template
    ) == common.ProblemSize(
        common.MatmulSize(32, 32, 1024, 1),
        common.ShapedType([1, 32, 1024], tuner_ctx.type.f32),
        common.ShapedType([1, 1024, 32], tuner_ctx.type.f32),
        common.ShapedType([1, 32, 32], tuner_ctx.type.f32),
        dispatch_parser.DispatchKind.batch_matmul,
    )


def test_get_shapes_batch_mmt(tuner_ctx: common.TunerContext) -> None:
    template = [
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {',
        r"flow.dispatch.tensor.store %21, %10, offsets = [0, 0, 0], sizes = [2, 4096, 640], strides = [1, 1, 1] : tensor<2x4096x640xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x4096x640xf16>>",
    ]
    assert dispatch_parser.BatchMmtParser().get_shapes(template) == common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 640, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i32),
        dispatch_parser.DispatchKind.batch_mmt,
    )


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
