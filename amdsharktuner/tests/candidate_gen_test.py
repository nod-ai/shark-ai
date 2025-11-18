# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest candidate_gen_test.py
"""

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu, transform  # type: ignore

from amdsharktuner import candidate_gen, common

from amdsharktuner.test_utils import tuner_ctx


def walk_collect_ops(
    op: ir.Operation,
    ops: list[ir.Operation],
    fn,
) -> ir.WalkResult:
    if fn(op):
        ops.append(op)
    return ir.WalkResult.ADVANCE


def get_ops_from_module(module: ir.Module, fn):
    ops: list[ir.Operation] = []
    for op in module.body.operations:
        op.walk(
            lambda op: walk_collect_ops(op, ops, fn),
            ir.WalkOrder.POST_ORDER,
        )
    return ops


def test_get_td_spec_contraction(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<2048x2048xf16>, %arg1: tensor<2048x2048xf16>) -> tensor<2048x2048xf32> {
                %cst = arith.constant 0.000000e+00 : f32
                %0 = tensor.empty() : tensor<2048x2048xf32>
                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
                %2 = linalg.generic {
                    indexing_maps = [
                        affine_map<(d0, d1, d2) -> (d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel", "reduction"]}
                    {root_op}
                    ins(%arg0, %arg1 : tensor<2048x2048xf16>, tensor<2048x2048xf16>)
                    outs(%1 : tensor<2048x2048xf32>) {
                ^bb0(%in: f16, %in_0: f16, %out: f32):
                    %3 = arith.extf %in : f16 to f32
                    %4 = arith.extf %in_0 : f16 to f32
                    %5 = arith.mulf %3, %4 : f32
                    %6 = arith.addf %out, %5 : f32
                    linalg.yield %6 : f32
                } -> tensor<2048x2048xf32>
                return %2 : tensor<2048x2048xf32>
            }
        }"""

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[8, 8, 0],
        reduction=[0, 0, 8],
        subgroup_basis=[[16, 16, 1], [0, 1, 2]],
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
    config_dict = common.get_translation_info_config(pipeline_options, waves_per_eu=8)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 16, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    ir_module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(ir_module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]

    tuner = candidate_gen.ContractionOpInterfaceTuner(root_op, tuner_ctx)
    td_spec_module = tuner.get_td_spec(
        [common.TuningConfiguration("compilation_info", compilation_info)]
    )
    assert td_spec_module

    named_sequence_ops: list[transform.NamedSequenceOp] = get_ops_from_module(
        module=td_spec_module,
        fn=lambda op: isinstance(op.opview, transform.NamedSequenceOp),
    )
    apply_config_sequence = None
    matcher_sequence = None
    entry_point = None
    for op in named_sequence_ops:
        if str(op.opview.sym_name) == '"apply_op_config"':
            apply_config_sequence = op
        elif str(op.opview.sym_name) == '"__kernel_config"':
            entry_point = op
        else:
            matcher_sequence = op

    assert apply_config_sequence
    assert matcher_sequence
    assert entry_point
    matcher_sequence_str = str(matcher_sequence)

    assert (
        "mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>" in matcher_sequence_str
    )
    assert "subgroup_basis = [[16, 16, 1], [0, 1, 2]]" in matcher_sequence_str
    assert "pipeline = LLVMGPUVectorDistribute" in matcher_sequence_str
    assert "workgroup_size = [16, 16, 1]" in matcher_sequence_str
    assert "subgroup_size = 16" in matcher_sequence_str
    assert "workgroup = [8, 8, 0]" in matcher_sequence_str
    assert "reduction = [0, 0, 8]" in matcher_sequence_str
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>"
        in matcher_sequence_str
    )
    assert 'llvm_func_attrs = {"amdgpu-waves-per-eu" = "8"}' in matcher_sequence_str


def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<2x34x34x2048xi8>, %arg1: tensor<3x3x2048x2048xi8>) -> tensor<2x32x32x2048xi32> {
                %cst = arith.constant 0 : i32
                %0 = tensor.empty() : tensor<2x32x32x2048xi32>
                %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x32x32x2048xi32>) -> tensor<2x32x32x2048xi32>
                %2 = linalg.conv_2d_nhwc_hwcf {root_op}
                    ins(%arg0, %arg1 : tensor<2x34x34x2048xi8>, tensor<3x3x2048x2048xi8>)
                    outs(%1 : tensor<2x32x32x2048xi32>) -> tensor<2x32x32x2048xi32>
                return %2 : tensor<2x32x32x2048xi32>
            }
        }"""

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 1, 464, 320, 0, 0, 0],
        reduction=[0, 0, 0, 0, 1, 1, 16],
        subgroup_basis=[[1, 1, 1, 1, 1, 1, 4], [0, 1, 2, 3, 4, 5, 6]],
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=False)
    config_dict = common.get_translation_info_config(pipeline_options, waves_per_eu=2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [256, 1, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    ir_module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(ir_module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    tuner = candidate_gen.ConvolutionOpInterfaceTuner(root_op, tuner_ctx)
    td_spec_module = tuner.get_td_spec(
        [common.TuningConfiguration("compilation_info", compilation_info)]
    )
    assert td_spec_module

    named_sequence_ops: list[transform.NamedSequenceOp] = get_ops_from_module(
        module=td_spec_module,
        fn=lambda op: isinstance(op.opview, transform.NamedSequenceOp),
    )
    apply_config_sequence = None
    matcher_sequence = None
    entry_point = None
    for op in named_sequence_ops:
        if str(op.opview.sym_name) == '"apply_op_config"':
            apply_config_sequence = op
        elif str(op.opview.sym_name) == '"__kernel_config"':
            entry_point = op
        else:
            matcher_sequence = op

    assert apply_config_sequence
    assert matcher_sequence
    assert entry_point

    matcher_sequence_str = str(matcher_sequence)

    assert (
        "mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>" in matcher_sequence_str
    )
    assert (
        "subgroup_basis = [[1, 1, 1, 1, 1, 1, 4], [0, 1, 2, 3, 4, 5, 6]]"
        in matcher_sequence_str
    )
    assert "pipeline = LLVMGPUVectorDistribute" in matcher_sequence_str
    assert "workgroup_size = [256, 1, 1]" in matcher_sequence_str
    assert "subgroup_size = 64" in matcher_sequence_str
    assert "workgroup = [1, 1, 464, 320, 0, 0, 0]" in matcher_sequence_str
    assert "reduction = [0, 0, 0, 0, 1, 1, 16]" in matcher_sequence_str
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
        in matcher_sequence_str
    )


def test_set_dispatch_tuner_with_matvec(tuner_ctx: common.TunerContext) -> None:
    # Make sure we do not crash on unsupported root ops (matvec).
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%A: tensor<8x224xf32>, %x: tensor<224xf32>) -> tensor<8xf32> {
                %init = tensor.empty() : tensor<8xf32>
                %y = linalg.matvec {root_op}
                    ins(%A, %x : tensor<8x224xf32>, tensor<224xf32>)
                    outs(%init : tensor<8xf32>) -> tensor<8xf32>
                return %y : tensor<8xf32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since mat-vec has invalid dimensions (M=[]).
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx)
    assert result is None


def test_set_dispatch_tuner_with_unsupported_conv(
    tuner_ctx: common.TunerContext,
) -> None:
    # Make sure we do not crash on unsupported conv layouts (nchw_fchw).
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<2x2048x34x34xi8>, %arg1: tensor<2048x2048x3x3xi8>) -> tensor<2x2048x32x32xi32> {
                %cst = arith.constant 0 : i32
                %0 = tensor.empty() : tensor<2x2048x32x32xi32>
                %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x2048x32x32xi32>) -> tensor<2x2048x32x32xi32>
                %2 = linalg.conv_2d_nchw_fchw {root_op}
                    ins(%arg0, %arg1 : tensor<2x2048x34x34xi8>, tensor<2048x2048x3x3xi8>)
                    outs(%1 : tensor<2x2048x32x32xi32>) -> tensor<2x2048x32x32xi32>
                return %2 : tensor<2x2048x32x32xi32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since conv with nchw_fchw layout is not supported.
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx)
    assert result is None


def test_set_dispatch_tuner_no_root_op(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
                %0 = linalg.add
                    ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>)
                    outs(%arg0 : tensor<256xf32>) -> tensor<256xf32>
                return %0 : tensor<256xf32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since no root_op is found.
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx)
    assert result is None


def test_set_dispatch_tuner_multiple_root_ops(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
                %0 = linalg.add {root_op}
                    ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>)
                    outs(%arg0 : tensor<256xf32>) -> tensor<256xf32>
                %1 = linalg.mul {root_op}
                    ins(%0, %0 : tensor<256xf32>, tensor<256xf32>)
                    outs(%0 : tensor<256xf32>) -> tensor<256xf32>
                return %1 : tensor<256xf32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since multiple root_ops are found.
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx)
    assert result is None
