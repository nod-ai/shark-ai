# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest dispatch_parser_test.py
"""

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func, iree_codegen, iree_gpu, linalg  # type: ignore

from amdsharktuner import common, dispatch_parser

from amdsharktuner.test_utils import tuner_ctx


GENERIC_TEMPLATE = r"""
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
        transpose_b_str = GENERIC_TEMPLATE.format(
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
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)

    with ir.Location.unknown():
        bmm_transposed_inputs_str = GENERIC_TEMPLATE.format(
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
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)

    with ir.Location.unknown():
        bmm_transposed_inputs_str = GENERIC_TEMPLATE.format(
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
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"


def test_get_matmul_named_op(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    with ir.Location.unknown(context):
        module = ir.Module.create()
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()

        with ir.InsertionPoint(module.body):
            a_type = ir.RankedTensorType.get((16, 64), f16)
            b_type = ir.RankedTensorType.get((64, 32), f16)
            c_type = ir.RankedTensorType.get((16, 32), f32)

            dim_m = ir.AffineDimExpr.get(0)
            dim_n = ir.AffineDimExpr.get(1)
            dim_k = ir.AffineDimExpr.get(2)
            a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
            b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
            c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])

            @func.FuncOp.from_py_func(a_type, b_type, c_type)
            def named_matmul(a, b, c):
                matmul_op = linalg.MatmulOp(
                    result_tensors=[c_type],
                    inputs=[a, b],
                    outputs=[c],
                    indexing_maps=[a_map, b_map, c_map],
                )
                matmul_op.operation.attributes["root_op"] = ir.UnitAttr.get()

        root_op_list = iree_codegen.get_tuner_root_ops(module)
        assert len(root_op_list) == 1, "Expected one root op"
        root_op = root_op_list[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        assert (
            dispatch_parser.get_parent_function_name(parser.get_root_op())
            == "named_matmul"
        )


def test_get_named_contraction_op(tuner_ctx: common.TunerContext):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()

        with ir.InsertionPoint(module.body):
            lhs_type = ir.RankedTensorType.get((5, 3), f32)
            rhs_type = ir.RankedTensorType.get((7, 3), f32)
            res_type = ir.RankedTensorType.get((5, 7), f32)

            @func.FuncOp.from_py_func(lhs_type, rhs_type, res_type)
            def named_contraction(lhs, rhs, res):
                dim_i = ir.AffineDimExpr.get(0)
                dim_j = ir.AffineDimExpr.get(1)
                dim_k = ir.AffineDimExpr.get(2)

                lhs_map = ir.AffineMap.get(3, 0, [dim_i, dim_k])
                rhs_map = ir.AffineMap.get(3, 0, [dim_j, dim_k])
                res_map = ir.AffineMap.get(3, 0, [dim_i, dim_j])

                contraction_op = linalg.ContractOp(
                    result_tensors=[res_type],
                    inputs=[lhs, rhs],
                    outputs=[res],
                    indexing_maps=[lhs_map, rhs_map, res_map],
                )
                contraction_op.attributes["root_op"] = ir.UnitAttr.get()

        root_op_list = iree_codegen.get_tuner_root_ops(module)
        assert len(root_op_list) == 1
        root_op = root_op_list[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        assert (
            dispatch_parser.get_parent_function_name(parser.get_root_op())
            == "named_contraction"
        )


def test_get_conv_nhwc_hwcf_operation(tuner_ctx: common.TunerContext) -> None:
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
    parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"
    assert (
        parser.has_valid_root_op()
    ), f"ConvolutionOpInterfaceParser does not support the op: {root_op.name}"


def test_get_group_conv_operation(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
    module {
      func.func @test(%arg0: tensor<2x10x10x7x4xf32>, %arg1: tensor<7x16x3x3x4xf32>, %arg2: tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32> {
        %0 = linalg.conv_2d_nhwgc_gfhwc {
           root_op,
           dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
        } ins(%arg0, %arg1: tensor<2x10x10x7x4xf32>, tensor<7x16x3x3x4xf32>)
          outs(%arg2: tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32>
        return %0 : tensor<2x8x8x7x16xf32>
      }
    }
    """
    module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"
    assert parser.has_valid_root_op() is False, "group convs aren't supported yet"


def test_get_generic_conv_operation(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    with ir.Location.name("generic_conv"):
        # nhwc_hwcf
        module_str = GENERIC_TEMPLATE.format(
            lhs_type=ir.RankedTensorType.get([2, 7, 7, 32], ir.F16Type.get()),
            rhs_type=ir.RankedTensorType.get([3, 3, 32, 64], ir.F16Type.get()),
            res_type=ir.RankedTensorType.get([2, 5, 5, 64], ir.F32Type.get()),
            lhs_map="affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>",
            rhs_map="affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>",
            res_map="affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>",
            iterator_types='["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]',
        )
    module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"
    assert parser.has_valid_root_op()


def test_get_mmt_tile_sizes(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[128, 320, 0],
        reduction=[0, 0, 32],
        subgroup_basis=[[1, 4, 1], [0, 1, 2]],
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
        subgroup_basis=[[1, 1, 1, 1, 1, 1, 4], [0, 1, 2, 3, 4, 5, 6]],
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


def test_get_attention_operation(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = r"""
        builtin.module  {
        func.func @attention_20x4096x64x4096x64(
        %q : tensor<20x4096x64xf16>,
        %k : tensor<20x4096x64xf16>,
        %v : tensor<20x4096x64xf16>,
        %scale : f16,
        %output : tensor<20x4096x64xf16>
    ) -> tensor<20x4096x64xf16> {
            %result = iree_linalg_ext.attention { root_op,
                indexing_maps = [
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                affine_map<(d0, d1, d2, d3, d4) -> ()>,
                affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
                ]
            } ins(%q, %k, %v, %scale : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16)
                outs(%output : tensor<20x4096x64xf16>) {
            ^bb0(%score: f32):
                iree_linalg_ext.yield %score : f32
            } -> tensor<20x4096x64xf16>
            return %result : tensor<20x4096x64xf16>
        }
    }
    """
    module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]

    parser = dispatch_parser.AttentionOpInterfaceParser(root_op, tuner_ctx)
    assert (
        dispatch_parser.get_parent_function_name(parser.get_root_op())
        == "attention_20x4096x64x4096x64"
    )
    assert parser.has_valid_root_op()

    indexing_maps_attr = root_op.attributes["indexing_maps"]
    affine_maps = [attr.value for attr in indexing_maps_attr]
    q_map = affine_maps[0]
    k_map = affine_maps[1]
    v_map = affine_maps[2]
    o_map = affine_maps[-1]
    result = iree_codegen.get_attention_op_detail(q_map, k_map, v_map, o_map)

    assert result.domain_rank == 5
    assert result.batch_dims == [0]
    assert result.m_dims == [1]
    assert result.k1_dims == [2]
    assert result.k2_dims == [3]
    assert result.n_dims == [4]
