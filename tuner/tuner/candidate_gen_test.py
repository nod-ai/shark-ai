# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest candidate_gen_test.py
"""

import pytest

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
from iree.compiler.dialects import transform  # type: ignore

from . import candidate_gen
from . import common
from . import op_matchers


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    from logging import Logger
    from unittest.mock import MagicMock

    with ir.Context() as ctx:
        logger: Logger = MagicMock(spec=Logger)
        yield common.TunerContext(ctx, logger)


def remove_comments(mlir: str) -> str:
    return "\n".join(
        filter(lambda x: not x.lstrip().startswith("//"), mlir.splitlines())
    )


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
        subgroup_m_count=16,
        subgroup_n_count=16,
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

    tuner = candidate_gen.ContractionOpInterfaceTuner()
    td_spec_module = tuner.get_td_spec(ir_module, compilation_info)
    assert td_spec_module

    named_sequence_ops: list[
        transform.NamedSequenceOp
    ] = op_matchers.get_ops_from_module(
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
    assert "subgroup_m_count = 16" in matcher_sequence_str
    assert "subgroup_n_count = 16" in matcher_sequence_str
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
        subgroup_m_count=1,
        subgroup_n_count=4,
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

    tuner = candidate_gen.ConvolutionOpInterfaceTuner()
    td_spec_module = tuner.get_td_spec(ir_module, compilation_info)
    assert td_spec_module

    named_sequence_ops: list[
        transform.NamedSequenceOp
    ] = op_matchers.get_ops_from_module(
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
    assert "subgroup_m_count = 1" in matcher_sequence_str
    assert "subgroup_n_count = 4" in matcher_sequence_str
    assert "pipeline = LLVMGPUVectorDistribute" in matcher_sequence_str
    assert "workgroup_size = [256, 1, 1]" in matcher_sequence_str
    assert "subgroup_size = 64" in matcher_sequence_str
    assert "workgroup = [1, 1, 464, 320, 0, 0, 0]" in matcher_sequence_str
    assert "reduction = [0, 0, 0, 0, 1, 1, 16]" in matcher_sequence_str
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
        in matcher_sequence_str
    )


def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [16, 16] subgroup_size = 16,",
        "<workgroup = [8, 8, 8], reduction = [8, 8, 8]>",
        "gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = None>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    M, N, K = 2048, 1280, 1280

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[8, 8, 0],
        reduction=[0, 0, 8],
        subgroup_m_count=16,
        subgroup_n_count=16,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
    config_dict = common.get_translation_info_config(pipeline_options, 8)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 16, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    problem_size = common.ProblemSize(
        common.MatmulSize(M, N, K),
        common.ShapedType([M, K], tuner_ctx.type.f16),
        common.ShapedType([N, K], tuner_ctx.type.f16),
        common.ShapedType([M, N], tuner_ctx.type.f32),
        common.DispatchKind.mmt,
    )
    tf_mlir = candidate_gen.MmtTuner().apply_params(
        problem_size, mlir_template, compilation_info
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    modified = remove_comments(modified)
    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 16, subgroup_n_count = 16"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [16, 16, 1] subgroup_size = 16"
        in modified
    )
    assert "workgroup = [8, 8, 0]" in modified
    assert "reduction = [0, 0, 8]" in modified
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>"
        in modified
    )
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "8"}' in modified


def test_apply_params_conv(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 16, subgroup_n_count = 16>",
        "<LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64,",
        "<workgroup = [1, 1, 64, 128, 1, 1, 32], reduction = [1, 1, 64, 128, 1, 1, 32]>",
        'gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}',
    ]

    n, oh, ow, oc, fh, fw, ic = 2, 64, 64, 640, 3, 3, 16

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[n, oh, ow, oc, fh, fw, 0],
        reduction=[0, 0, 0, 0, 0, 0, ic],
        subgroup_m_count=1,
        subgroup_n_count=4,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get(
        reorder_workgroups_strategy=iree_gpu.ReorderWorkgroupsStrategyAttr.get(
            iree_gpu.ReorderWorkgroupsStrategy.Transpose
        )
    )
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [256, 1, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    problem_size = common.ProblemSize(
        common.MatmulSize(oh * ow, oc, fh * fw * ic),
        common.ShapedType([n, oh + 2, ow + 2, oc], tuner_ctx.type.f16),
        common.ShapedType([fh, fw, ic, oc], tuner_ctx.type.f16),
        common.ShapedType([n, oh, ow, oc], tuner_ctx.type.f32),
        common.DispatchKind.conv,
    )
    tf_mlir = candidate_gen.ConvTuner().apply_params(
        problem_size, mlir_template, compilation_info
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    modified = remove_comments(modified)
    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 4"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in modified
    )
    assert "workgroup = [2, 64, 64, 640, 3, 3, 0]" in modified
    assert "reduction = [0, 0, 0, 0, 0, 0, 16]" in modified
    assert (
        "gpu_pipeline_options = #iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>"
        in modified
    )
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_contract(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64,",
        "<workgroup = [1, 1, 1, 64, 64, 128], reduction = [1, 1, 1, 64, 64, 128]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "*mnk"
    problem_size = common.ProblemSize(
        common.MatmulSize(2048, 3840, 1280),
        common.ShapedType([2, 1024, 1280], tuner_ctx.type.f16),
        common.ShapedType([3, 20, 64, 1280], tuner_ctx.type.f16),
        common.ShapedType([3, 2, 20, 1024, 64], tuner_ctx.type.f32),
        common.DispatchKind.contraction,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 480, 384, 0],
        reduction=[0, 0, 0, 32],
        subgroup_m_count=1,
        subgroup_n_count=4,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [256, 1, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    tf_mlir = candidate_gen.ContractionTuner("mk", "nk", tile_dims).apply_params(
        problem_size, mlir_template, compilation_info
    )

    new_mlir = tf_mlir.modified

    assert new_mlir
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 1, subgroup_n_count = 4"
        in new_mlir
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [256, 1, 1] subgroup_size = 64"
        in new_mlir
    )
    assert "workgroup = [1, 480, 384, 0]" in new_mlir
    assert "reduction = [0, 0, 0, 32]" in new_mlir
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in new_mlir


def test_apply_params_batch_matmul(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<workgroup = [1, 128, 64, 64], reduction = [1, 128, 64, 64]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    tile_dims = "bmnk"
    problem_size = common.ProblemSize(
        common.MatmulSize(968, 320, 640, 64),
        common.ShapedType([64, 968, 640], tuner_ctx.type.f16),
        common.ShapedType([64, 640, 320], tuner_ctx.type.f16),
        common.ShapedType([64, 968, 320], tuner_ctx.type.f32),
        common.DispatchKind.batch_matmul,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 416, 320, 0],
        reduction=[0, 0, 0, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [128, 2, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    tf_mlir = candidate_gen.BatchMatmulTuner("mk", "nk", tile_dims).apply_params(
        problem_size, mlir_template, compilation_info
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    modified = remove_comments(modified)

    assert embeddable
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "workgroup = [1, 416, 320, 0]" in modified
    assert "reduction = [0, 0, 0, 128]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_float(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<workgroup = [1, 128, 128, 64], reduction = [1, 128, 128, 64]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.f16),
        common.ShapedType([2, 640, 640], tuner_ctx.type.f16),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.f32),
        common.DispatchKind.batch_mmt,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 128, 64, 0],
        reduction=[0, 0, 0, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [128, 2, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    tf_mlir = candidate_gen.BatchMmtTuner().apply_params(
        problem_size, mlir_template, compilation_info
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert embeddable
    assert modified
    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "workgroup = [1, 128, 64, 0]" in modified
    assert "reduction = [0, 0, 0, 128]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}' in modified


def test_apply_params_batch_mmt_int(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<workgroup = [1, 128, 128, 64], reduction = [1, 128, 128, 64]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 640, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i32),
        common.DispatchKind.batch_mmt,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 128, 64, 0],
        reduction=[0, 0, 0, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 4)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [128, 2, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    tf_mlir = candidate_gen.BatchMmtTuner().apply_params(
        problem_size, mlir_template, compilation_info
    )

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    assert "//   transform.named_sequence @match_batch_mmt_2x4096x640x640(" in modified
    modified = remove_comments(modified)

    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "workgroup = [1, 128, 64, 0]" in modified
    assert "reduction = [0, 0, 0, 128]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in modified

    assert embeddable
    assert "transform.named_sequence @match_op(" in embeddable
    assert (
        "transform.include @match_batch_mmt_i8_i8_i32 failures(propagate)" in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %lhs = tensor<2x4096x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %rhs = tensor<2x640x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "%config = transform.param.constant #iree_codegen.compilation_info<"
        in embeddable
    )
    assert "workgroup = [1, 128, 64, 0]" in embeddable
    assert "reduction = [0, 0, 0, 128]" in embeddable
    assert 'llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in embeddable
    assert "workgroup_size = [128, 2, 1] subgroup_size = 64" in embeddable


def test_apply_params_broadcast_rhs_mmt(tuner_ctx: common.TunerContext) -> None:
    mlir_template = [
        "<intrinsic = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>, subgroup_m_count = 4, subgroup_n_count = 1>}>",
        "<LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64,",
        "<workgroup = [1, 128, 128, 64]], reduction = [1, 128, 128, 64]>",
        '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}',
    ]

    problem_size = common.ProblemSize(
        common.MatmulSize(4096, 640, 640, 2),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i8),
        common.ShapedType([640, 640], tuner_ctx.type.i8),
        common.ShapedType([2, 4096, 640], tuner_ctx.type.i32),
        common.DispatchKind.broadcast_rhs_mmt,
    )

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 128, 64, 0],
        reduction=[0, 0, 0, 128],
        subgroup_m_count=2,
        subgroup_n_count=2,
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 4)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [128, 2, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    tf_mlir = candidate_gen.ContractionTuner(
        "mk", "nk", "mnk"
    ).apply_params_broadcast_rhs_mmt(problem_size, mlir_template, compilation_info)

    modified = tf_mlir.modified
    embeddable = tf_mlir.embeddable

    assert modified
    assert (
        "//   transform.named_sequence @match_broadcast_rhs_mmt_Bx4096x640x640("
        in modified
    )
    modified = remove_comments(modified)

    assert (
        "intrinsic = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, subgroup_m_count = 2, subgroup_n_count = 2"
        in modified
    )
    assert (
        "LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64"
        in modified
    )
    assert "workgroup = [1, 128, 64, 0]" in modified
    assert "reduction = [0, 0, 0, 128]" in modified
    assert '{llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in modified

    assert embeddable
    assert "transform.named_sequence @match_op(" in embeddable
    assert (
        "transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate)"
        in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %lhs = tensor<?x4096x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "transform.iree.match.cast_compatible_type %rhs = tensor<640x640xi8> : !transform.any_value"
        in embeddable
    )
    assert (
        "%config = transform.param.constant #iree_codegen.compilation_info<"
        in embeddable
    )
    assert "workgroup = [1, 128, 64, 0]" in embeddable
    assert "reduction = [0, 0, 0, 128]" in embeddable
    assert 'llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}' in embeddable
    assert "workgroup_size = [128, 2, 1] subgroup_size = 64" in embeddable


def test_detect_broadcast_rhs_mmt(tuner_ctx: common.TunerContext) -> None:
    mlir_lines = [
        r"%18 = tensor.empty() : tensor<2x1024x10240xi32>",
        r"%19 = linalg.fill {lowering_config = #iree_codegen.lowering_config<workgroup = [[1, 64, 128, 0]], reduction = [[0, 0, 0, 128]]>} ins(%c0_i32 : i32) outs(%18 : tensor<2x1024x10240xi32>) -> tensor<2x1024x10240xi32>",
        r'%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%11, %12 : tensor<2x1024x1280xi8>, tensor<10240x1280xi8>) outs(%19 : tensor<2x1024x10240xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128, 128]]>} {',
    ]
    assert candidate_gen.ContractionTuner("mk", "nk", "mnk").is_broadcast_rhs_mmt(
        mlir_lines
    )
