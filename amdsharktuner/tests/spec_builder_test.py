# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest spec_builder_test.py
"""

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import arith, func, iree_codegen, iree_gpu, linalg  # type: ignore

from amdsharktuner import common, dispatch_parser, spec_builder

from amdsharktuner.test_utils import tuner_ctx


def create_generic_module(tuner_ctx: common.TunerContext) -> ir.Module:
    ctx = tuner_ctx.mlir_ctx
    with ir.Location.unknown(ctx):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f16 = ir.F16Type.get()
            f32 = ir.F32Type.get()

            lhs_type = ir.RankedTensorType.get([1024, 512], f16)
            rhs_type = ir.RankedTensorType.get([2048, 512], f16)
            output_type = ir.RankedTensorType.get([1024, 2048], f32)

            dim0 = ir.AffineDimExpr.get(0)
            dim1 = ir.AffineDimExpr.get(1)
            dim2 = ir.AffineDimExpr.get(2)

            a_map = ir.AffineMap.get(3, 0, [dim0, dim2])
            b_map = ir.AffineMap.get(3, 0, [dim1, dim2])
            c_map = ir.AffineMap.get(3, 0, [dim0, dim1])

            indexing_maps = ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(a_map),
                    ir.AffineMapAttr.get(b_map),
                    ir.AffineMapAttr.get(c_map),
                ]
            )

            iterator_types_attr = ir.ArrayAttr.get(
                [
                    ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                    ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                    ir.Attribute.parse("#linalg.iterator_type<reduction>"),
                ]
            )

            @func.FuncOp.from_py_func(lhs_type, rhs_type, output_type)
            def matmul_func(arg0, arg1, arg2):
                generic_op = linalg.GenericOp(
                    result_tensors=[output_type],
                    inputs=[arg0, arg1],
                    outputs=[arg2],
                    indexing_maps=indexing_maps,
                    iterator_types=iterator_types_attr,
                )
                generic_op.operation.attributes["root_op"] = ir.UnitAttr.get()

                block = generic_op.regions[0].blocks.append(f16, f16, f32)
                with ir.InsertionPoint(block):
                    ext0 = arith.ExtFOp(f32, block.arguments[0]).result
                    ext1 = arith.ExtFOp(f32, block.arguments[1]).result
                    mul = arith.MulFOp(ext0, ext1)
                    add = arith.AddFOp(block.arguments[2], mul)
                    linalg.YieldOp([add])

        return module


def create_batch_matmul_module(tuner_ctx: common.TunerContext) -> ir.Module:
    ctx = tuner_ctx.mlir_ctx
    with ir.Location.unknown(ctx):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f16 = ir.F16Type.get()
            f32 = ir.F32Type.get()

            lhs_type = ir.RankedTensorType.get([4, 1024, 512], f16)
            rhs_type = ir.RankedTensorType.get([4, 2048, 512], f16)
            output_type = ir.RankedTensorType.get([4, 1024, 2048], f32)

            dim0 = ir.AffineDimExpr.get(0)
            dim1 = ir.AffineDimExpr.get(1)
            dim2 = ir.AffineDimExpr.get(2)
            dim3 = ir.AffineDimExpr.get(3)

            a_map = ir.AffineMap.get(4, 0, [dim0, dim1, dim3])
            b_map = ir.AffineMap.get(4, 0, [dim0, dim2, dim3])
            c_map = ir.AffineMap.get(4, 0, [dim0, dim1, dim2])

            indexing_maps = ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(a_map),
                    ir.AffineMapAttr.get(b_map),
                    ir.AffineMapAttr.get(c_map),
                ]
            )

            iterator_types_attr = ir.ArrayAttr.get(
                [
                    ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                    ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                    ir.Attribute.parse("#linalg.iterator_type<parallel>"),
                    ir.Attribute.parse("#linalg.iterator_type<reduction>"),  # K
                ]
            )

            @func.FuncOp.from_py_func(lhs_type, rhs_type, output_type)
            def batch_matmul_func(arg0, arg1, arg2):
                generic_op = linalg.GenericOp(
                    result_tensors=[output_type],
                    inputs=[arg0, arg1],
                    outputs=[arg2],
                    indexing_maps=indexing_maps,
                    iterator_types=iterator_types_attr,
                )
                generic_op.operation.attributes["root_op"] = ir.UnitAttr.get()

                block = generic_op.regions[0].blocks.append(f16, f16, f32)
                with ir.InsertionPoint(block):
                    ext0 = arith.ExtFOp(f32, block.arguments[0]).result
                    ext1 = arith.ExtFOp(f32, block.arguments[1]).result
                    mul = arith.MulFOp(ext0, ext1)
                    add = arith.AddFOp(block.arguments[2], mul)
                    linalg.YieldOp([add])

        return module


def test_spec_builder(tuner_ctx: common.TunerContext) -> None:
    module = create_generic_module(tuner_ctx)
    root_ops = iree_codegen.get_tuner_root_ops(module)
    assert len(root_ops) == 1, "Expected exactly one root op"
    root_op = root_ops[0]

    attributes = ir.DictAttr.get({"reduction": ir.ArrayAttr.get([])})
    lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.None_
    )
    translation_info = iree_codegen.TranslationInfoAttr.get(pipeline_attr)
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
    op_info = parser.get_op_info()

    builder = spec_builder.ContractionSpecBuilder(op_info)
    spec_module = builder.build_td_spec(
        tuner_ctx,
        [
            common.TuningConfiguration(
                name="compilation_info", configuration=compilation_info
            )
        ],
    )
    assert spec_module
    assert isinstance(spec_module, ir.Module)
    spec_str = str(spec_module)
    assert "@match_matmul_func -> @apply_op_config" in spec_str
    assert 'transform.annotate %arg0 "compilation_info" = %arg1' in spec_str
    assert "transform.iree.match.contraction" in spec_str
    assert "lhs_type =" in spec_str
    assert "rhs_type =" in spec_str
    assert "output_type =" in spec_str
    assert "transform.iree.match.dims_equal %batch_dims, []" in spec_str
    assert "transform.iree.match.dims_equal %m_dims, [1024]" in spec_str
    assert "transform.iree.match.dims_equal %n_dims, [2048]" in spec_str
    assert "transform.iree.match.dims_equal %k_dims, [512]" in spec_str

    qk_config = iree_gpu.LoweringConfigAttr.get(
        ir.DictAttr.get(
            {
                "workgroup": ir.ArrayAttr.get(
                    [
                        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
                        for i in [1, 1, 1]
                    ]
                )
            }
        )
    )
    pv_config = iree_gpu.LoweringConfigAttr.get(
        ir.DictAttr.get(
            {
                "workgroup": ir.ArrayAttr.get(
                    [
                        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)
                        for i in [2, 2, 2]
                    ]
                )
            }
        )
    )
    decomposition_config = ir.DictAttr.get(
        {
            "qk_attrs": ir.DictAttr.get({"attention_qk_matmul": qk_config}),
            "pv_attrs": ir.DictAttr.get({"attention_pv_matmul": pv_config}),
        }
    )
    config_list = [
        common.TuningConfiguration(
            name="compilation_info", configuration=compilation_info
        ),
        common.TuningConfiguration(
            name="decomposition_config", configuration=decomposition_config
        ),
    ]

    builder = spec_builder.ContractionSpecBuilder(op_info)
    spec_module = builder.build_td_spec(tuner_ctx, config_list)
    assert spec_module
    assert isinstance(spec_module, ir.Module)
    spec_str = str(spec_module)
    assert "@match_matmul_func -> @apply_op_config" in spec_str
    assert 'transform.annotate %arg0 "compilation_info" = %arg1' in spec_str
    assert 'transform.annotate %arg0 "decomposition_config" = %arg2' in spec_str


def test_spec_builder_with_batch_dims(tuner_ctx: common.TunerContext) -> None:
    module = create_batch_matmul_module(tuner_ctx)
    root_ops = iree_codegen.get_tuner_root_ops(module)
    assert len(root_ops) == 1, "Expected exactly one root op"
    root_op = root_ops[0]

    attributes = ir.DictAttr.get({"reduction": ir.ArrayAttr.get([])})
    lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.None_
    )
    translation_info = iree_codegen.TranslationInfoAttr.get(pipeline_attr)
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
    op_info = parser.get_op_info()

    builder = spec_builder.ContractionSpecBuilder(op_info)
    spec_module = builder.build_td_spec(
        tuner_ctx,
        [
            common.TuningConfiguration(
                name="compilation_info", configuration=compilation_info
            )
        ],
    )
    assert spec_module
    assert isinstance(spec_module, ir.Module)
    spec_str = str(spec_module)

    assert "@match_batch_matmul_func -> @apply_op_config" in spec_str
    assert 'transform.annotate %arg0 "compilation_info" = %arg1' in spec_str
    assert "transform.iree.match.contraction" in spec_str
    assert "lhs_type =" in spec_str
    assert "rhs_type =" in spec_str
    assert "output_type =" in spec_str

    assert "transform.iree.match.dims_equal %batch_dims, [4]" in spec_str
    assert "transform.iree.match.dims_equal %m_dims, [1024]" in spec_str
    assert "transform.iree.match.dims_equal %n_dims, [2048]" in spec_str
    assert "transform.iree.match.dims_equal %k_dims, [512]" in spec_str

    assert "lhs_type = f16" in spec_str
    assert "rhs_type = f16" in spec_str
    assert "output_type = f32" in spec_str
