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

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import sharktuner
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
from iree.compiler.dialects import func, linalg  # type: ignore

from sharktuner import common
from sharktuner import constraint_generator
from sharktuner import dispatch_constraints

from sharktuner.test_utils import tuner_ctx


def build_func_with_matmul(
    module: ir.Module,
    m: int,
    n: int,
    k: int,
    lhs_type: ir.Type,
    rhs_type: ir.Type,
    res_type: ir.Type,
) -> None:
    a_type = ir.RankedTensorType.get((m, k), lhs_type)
    b_type = ir.RankedTensorType.get((k, n), rhs_type)
    c_type = ir.RankedTensorType.get((m, n), res_type)

    dim_m = ir.AffineDimExpr.get(0)
    dim_n = ir.AffineDimExpr.get(1)
    dim_k = ir.AffineDimExpr.get(2)
    a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
    b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
    c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(a_type, b_type, c_type)
        def named_matmul(a: ir.Value, b: ir.Value, c: ir.Value) -> None:
            matmul_op = linalg.MatmulOp(
                result_tensors=[c_type],
                inputs=[a, b],
                outputs=[c],
                indexing_maps=[a_map, b_map, c_map],
            )
            matmul_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def build_func_with_conv2d_nhwc_hwcf(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    kernel_type: ir.Type,
    output_type: ir.Type,
) -> None:
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    kernel_tensor_type = ir.RankedTensorType.get(kernel_shape, kernel_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, kernel_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            conv_op = linalg.Conv2DNhwcHwcfOp(
                inputs=[arg0, arg1],
                outputs=[arg2],
                result_tensors=[output_tensor_type],
            )
            conv_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def test_generate_solutions(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 2048, 3840, 1280
    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        gen = constraint_generator.ContractionOpInterfaceConstraintGenerator(root_op)

        assert gen.dims.batch == []
        assert gen.dims.m == [0]
        assert gen.dims.n == [1]
        assert gen.dims.k == [2]

        assert gen.matmul_size.B == []
        assert gen.matmul_size.M == [2048]
        assert gen.matmul_size.N == [3840]
        assert gen.matmul_size.K == [1280]

        assert gen.lhs_type.shape == [2048, 1280]
        assert gen.rhs_type.shape == [1280, 3840]
        assert gen.res_type.shape == [2048, 3840]

        configs = gen.generate_solutions(
            tuner_context=tuner_ctx,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            num_subgroups=4,
            mma_intrinsics=[
                iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
                iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
                iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
                iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
            ],
            pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
        )

        assert list(configs), "Expected at least one valid solution"


def test_generate_solutions_tile_and_fuse_contraction_padding(
    tuner_ctx: common.TunerContext,
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 5369, 112, 112

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        gen = constraint_generator.ContractionOpInterfaceConstraintGenerator(root_op)

        assert gen.dims.batch == []
        assert gen.dims.m == [0]
        assert gen.dims.n == [1]
        assert gen.dims.k == [2]

        assert gen.matmul_size.M == [5369]
        assert gen.matmul_size.N == [112]
        assert gen.matmul_size.K == [112]

        assert gen.lhs_type.shape == [5369, 112]
        assert gen.rhs_type.shape == [112, 112]
        assert gen.res_type.shape == [5369, 112]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                num_subgroups=4,
                mma_intrinsics=[
                    iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
                ],
                allowed_waves_per_eu=[2],
                pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
            )
        )

        assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
        assert all(
            isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions
        )

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
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    input_shape = (2, 7, 7, 32)
    kernel_shape = (3, 3, 32, 64)
    output_shape = (2, 5, 5, 64)

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_hwcf(
            module=module,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            input_type=f16,
            kernel_type=f16,
            output_type=f32,
        )

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        gen = constraint_generator.ConvolutionOpInterfaceConstraintGenerator(root_op)

        assert gen.dims.batch == []
        assert gen.dims.m == [0, 1, 2]
        assert gen.dims.n == [3]
        assert gen.dims.k == [4, 5, 6]

        assert gen.matmul_size.B == []
        assert gen.matmul_size.M == [2, 5, 5]
        assert gen.matmul_size.N == [64]
        assert gen.matmul_size.K == [3, 3, 32]

        assert gen.lhs_type.shape == [2, 7, 7, 32]
        assert gen.rhs_type.shape == [3, 3, 32, 64]
        assert gen.res_type.shape == [2, 5, 5, 64]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                num_subgroups=4,
                mma_intrinsics=[iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16],
            )
        )

        assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
        assert all(
            isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions
        )
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
