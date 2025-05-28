# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest constraint_generator_test.py
"""

import pytest
import z3  # type: ignore

from typing import Generator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu, iree_codegen  # type: ignore

from sharktuner import common
from sharktuner import constraint_generator
from sharktuner import dispatch_constraints

from sharktuner.test_utils import tuner_ctx


def test_generate_solutions(tuner_ctx: common.TunerContext) -> None:
    matmul_size = common.ContractionSizes([2048], [3840], [1280])
    contraction_dims = common.ContractionDimensions([0], [1], [2])

    lhs_type = common.ShapedType([2048, 1280], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([3840, 1280], tuner_ctx.type.f16)
    res_type = common.ShapedType([2048, 3840], tuner_ctx.type.f32)

    configs = constraint_generator.generate_generic_contraction_solutions(
        tuner_ctx=tuner_ctx,
        contraction_dims=contraction_dims,
        matmul_size=matmul_size,
        lhs_type=lhs_type,
        rhs_type=rhs_type,
        res_type=res_type,
        dispatch_kind=common.DispatchKind.contraction,
        num_subgroups=4,
        mma_list=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
        pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    )
    assert list(configs), "Expected at least one valid solution"


def test_generate_solutions_tile_and_fuse_contraction_padding(
    tuner_ctx: common.TunerContext,
) -> None:
    matmul_size = common.ContractionSizes([5369], [112], [112])
    contraction_dims = common.ContractionDimensions([0], [1], [2])

    lhs_type = common.ShapedType([5369, 112], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([112, 112], tuner_ctx.type.f16)
    res_type = common.ShapedType([5369, 112], tuner_ctx.type.f32)

    mma_intrinsics = [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
    ]

    solutions = list(
        constraint_generator.generate_generic_contraction_solutions(
            tuner_ctx=tuner_ctx,
            contraction_dims=contraction_dims,
            matmul_size=matmul_size,
            lhs_type=lhs_type,
            rhs_type=rhs_type,
            res_type=res_type,
            dispatch_kind=common.DispatchKind.contraction,
            num_subgroups=4,
            mma_list=mma_intrinsics,
            allowed_waves_per_eu=[2],
            pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        )
    )

    assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
    assert all(isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions)

    assert all(
        "padding =" in str(sol.lowering_config) for sol in solutions
    ), "Not all lowering configs have padding option."

    assert all(
        [int(x) for x in sol.lowering_config.attributes["promote_operands"]]
        == [0, 1, 2]
        for sol in solutions
    ), "Not all lowering configs have promote_operands = [0, 1, 2]."


def test_generate_solutions_tile_and_fuse_conv_padding(
    tuner_ctx: common.TunerContext,
) -> None:
    contraction_dims = common.ContractionDimensions(
        batch=[],
        m=[0, 1, 2],
        n=[3],
        k=[4, 5, 6],
    )
    matmul_size = common.ContractionSizes(
        B=[],
        M=[2, 5, 5],
        N=[64],
        K=[3, 3, 32],
    )
    lhs_type = common.ShapedType([2, 7, 7, 32], tuner_ctx.type.f16)
    rhs_type = common.ShapedType([64, 3, 3, 32], tuner_ctx.type.f16)
    res_type = common.ShapedType([2, 5, 5, 64], tuner_ctx.type.f32)

    solutions = list(
        constraint_generator.generate_generic_contraction_solutions(
            tuner_ctx=tuner_ctx,
            contraction_dims=contraction_dims,
            matmul_size=matmul_size,
            lhs_type=lhs_type,
            rhs_type=rhs_type,
            res_type=res_type,
            dispatch_kind=common.DispatchKind.conv,
            num_subgroups=4,
            mma_list=[iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16],
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        )
    )

    assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
    assert all(isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions)
    assert all(
        "padding =" in str(sol.lowering_config) for sol in solutions
    ), "Not all lowering configs have padding option"
    assert all(
        [int(x) for x in sol.lowering_config.attributes["promote_operands"]]
        == [0, 1, 2]
        for sol in solutions
    ), "Not all lowering configs have promote_operands = [0, 1, 2]"


def test_adjust_problem_size_for_pipeline(
    tuner_ctx: common.TunerContext,
) -> None:
    matmul_size = common.ContractionSizes(
        M=[32],
        N=[64],
        K=[128],
        B=[2],
    )
    contraction_dims = common.ContractionDimensions(
        m=[1],
        n=[2],
        k=[3],
        batch=[0],
    )
    taf_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
    pipeline_options_space = dispatch_constraints.PipelineOptionsSearchSpace(
        prefetch_shared_memory=[True],
        no_reduce_shared_memory_bank_conflicts=[True, False],
        use_igemm_convolution=[None],
    )

    constraint_generator.adjust_problem_size_for_pipeline(
        contraction_dims=contraction_dims,
        matmul_size=matmul_size,
        dispatch_kind=common.DispatchKind.contraction,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=taf_pipeline,
    )
    assert pipeline_options_space.use_igemm_convolution == [None]
    assert matmul_size.K == [128]
    assert contraction_dims.k == [3]

    conv_size = common.ContractionSizes(
        M=[2, 32, 32],
        N=[256],
        K=[3, 3, 512],
    )
    conv_dims = common.ContractionDimensions(
        m=[0, 1, 2],
        n=[3],
        k=[4, 5, 6],
    )
    vec_dist_pipeline = (
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    constraint_generator.adjust_problem_size_for_pipeline(
        contraction_dims=conv_dims,
        matmul_size=conv_size,
        dispatch_kind=common.DispatchKind.conv,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=vec_dist_pipeline,
    )
    assert pipeline_options_space.use_igemm_convolution == [None]
    assert conv_size.K == [3, 3, 512]
    assert conv_dims.k == [4, 5, 6]

    constraint_generator.adjust_problem_size_for_pipeline(
        contraction_dims=conv_dims,
        matmul_size=conv_size,
        dispatch_kind=common.DispatchKind.conv,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=taf_pipeline,
    )
    assert pipeline_options_space.use_igemm_convolution == [True]
    assert conv_size.K == [4608]
    assert conv_dims.k == [4]
