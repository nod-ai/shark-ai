# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest constraint_generator_test.py
"""

import pytest

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func, iree_codegen, iree_gpu, linalg  # type: ignore

from amdsharktuner import (
    common,
    constraint_generator,
    dispatch_constraints,
    dispatch_parser,
)

from amdsharktuner.test_utils import tuner_ctx


@pytest.fixture
def gpu_target_info(tuner_ctx: common.TunerContext) -> iree_gpu.TargetInfo:
    context = tuner_ctx.mlir_ctx
    return iree_gpu.TargetInfo(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[64],
        max_workgroup_sizes=[1024, 1024, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        workgroup_count=304,
        simds_per_workgroup=4,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )


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


def test_generate_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
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

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = constraint_generator.ContractionOpInterfaceConstraintGenerator(op_info)

        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0]
        assert gen.op_info.dims.n == [1]
        assert gen.op_info.dims.k == [2]

        assert gen.op_info.matmul_size.B == []
        assert gen.op_info.matmul_size.M == [2048]
        assert gen.op_info.matmul_size.N == [3840]
        assert gen.op_info.matmul_size.K == [1280]

        assert gen.op_info.lhs_type.shape == [2048, 1280]
        assert gen.op_info.rhs_type.shape == [1280, 3840]
        assert gen.op_info.res_type.shape == [2048, 3840]

        configs = gen.generate_solutions(
            tuner_context=tuner_ctx,
            gpu_target_info=gpu_target_info,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            num_subgroups=4,
            pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
        )

        assert list(configs), "Expected at least one valid solution"


def test_generate_attention_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    op_info = dispatch_parser.AttentionOpInfo(
        root_op=None,
        indexing_maps=[],
        domain_rank=5,
        batch_dims=[0],
        m_dims=[1],
        n_dims=[2],
        k1_dims=[3],
        k2_dims=[4],
        batch_sizes=[2],
        m_sizes=[64],
        n_sizes=[32],
        k1_sizes=[64],
        k2_sizes=[64],
        query_type=f16,
        key_type=f16,
        value_type=f16,
        output_type=f16,
        transposed_q=True,
        transposed_k=True,
        transposed_v=False,
        qk_matmul=common.MatmulShapeType(
            m=64,
            n=64,
            k=64,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f32,
        ),
        pv_matmul=common.MatmulShapeType(
            m=64,
            n=32,
            k=64,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f32,
        ),
    )

    solutions = list(
        constraint_generator.generate_attention_solutions(
            tuner_ctx=tuner_ctx,
            gpu_target_info=gpu_target_info,
            op_info=op_info,
            dispatch_kind=common.DispatchKind.attention,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            num_subgroups=4,
            pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
        )
    )

    assert len(solutions) > 0, "Expected at least one valid attention tuning solution"
    for config_list in solutions:
        assert len(config_list) == 2
        assert config_list[0].name == "compilation_info"
        assert config_list[1].name == "decomposition_config"
        assert isinstance(
            config_list[0].configuration, iree_codegen.CompilationInfoAttr
        )
        assert isinstance(config_list[1].configuration, ir.DictAttr)

        # Verify that prefetch_shared_memory is set based on layout matching.
        compilation_info = config_list[0].configuration
        translation_info = compilation_info.translation_info
        if translation_info.configuration:
            pipeline_options = translation_info.configuration[
                common.GPU_PIPELINE_OPTIONS_KEY
            ]
            # prefetch_shared_memory should be explicitly set to a boolean (not None).
            assert isinstance(
                pipeline_options.prefetch_shared_memory, bool
            ), "prefetch_shared_memory must be explicitly set to True or False"


def test_generate_solutions_tile_and_fuse_contraction_padding(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
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

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = constraint_generator.ContractionOpInterfaceConstraintGenerator(op_info)

        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0]
        assert gen.op_info.dims.n == [1]
        assert gen.op_info.dims.k == [2]

        assert gen.op_info.matmul_size.M == [5369]
        assert gen.op_info.matmul_size.N == [112]
        assert gen.op_info.matmul_size.K == [112]

        assert gen.op_info.lhs_type.shape == [5369, 112]
        assert gen.op_info.rhs_type.shape == [112, 112]
        assert gen.op_info.res_type.shape == [5369, 112]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                num_subgroups=4,
                allowed_waves_per_eu=[2],
                pipeline_options_search_space=dispatch_constraints.PipelineOptionsSearchSpace(),
            )
        )

        assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
        for solution in solutions:
            assert len(solution) == 1, f"Expected a single-item list, got: {solution}"
            config = solution[0]
            assert isinstance(
                config, common.TuningConfiguration
            ), f"Expected TuningConfiguration, got: {type(config)}"

            assert (
                config.name == "compilation_info"
            ), f"Expected key 'compilation_info', got: {config.name}"
            assert isinstance(
                config.configuration, iree_codegen.CompilationInfoAttr
            ), f"Expected CompilationInfoAttr, got: {type(config.configuration)}"

            lowering_config = config.configuration.lowering_config
            assert "padding =" in str(
                lowering_config
            ), f"Missing padding in lowering config: {lowering_config}"
            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
            assert promote == [0, 1, 2]


def test_generate_solutions_tile_and_fuse_conv_padding(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
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

        parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = constraint_generator.ConvolutionOpInterfaceConstraintGenerator(op_info)

        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0, 1, 2]
        assert gen.op_info.dims.n == [3]
        assert gen.op_info.dims.k == [4, 5, 6]

        assert gen.op_info.matmul_size.B == []
        assert gen.op_info.matmul_size.M == [2, 5, 5]
        assert gen.op_info.matmul_size.N == [64]
        assert gen.op_info.matmul_size.K == [3, 3, 32]

        assert gen.op_info.lhs_type.shape == [2, 7, 7, 32]
        assert gen.op_info.rhs_type.shape == [3, 3, 32, 64]
        assert gen.op_info.res_type.shape == [2, 5, 5, 64]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                num_subgroups=4,
            )
        )

        assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
        for solution in solutions:
            assert len(solution) == 1, f"Expected a single-item list, got: {solution}"
            config = solution[0]
            assert isinstance(
                config, common.TuningConfiguration
            ), f"Expected TuningConfiguration, got: {type(config)}"

            assert (
                config.name == "compilation_info"
            ), f"Expected key 'compilation_info', got: {config.name}"
            assert isinstance(
                config.configuration, iree_codegen.CompilationInfoAttr
            ), f"Expected CompilationInfoAttr, got: {type(config.configuration)}"

            lowering_config = config.configuration.lowering_config
            assert "padding =" in str(
                lowering_config
            ), f"Missing padding in lowering config: {lowering_config}"
            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
            assert promote == [0, 1, 2]


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
        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
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
    constraint_generator.adjust_problem_size_for_pipeline(
        contraction_dims=conv_dims,
        matmul_size=conv_size,
        dispatch_kind=common.DispatchKind.conv,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    )
    assert pipeline_options_space.use_igemm_convolution == [None]
    assert conv_size.K == [3, 3, 512]
    assert conv_dims.k == [4, 5, 6]

    constraint_generator.adjust_problem_size_for_pipeline(
        contraction_dims=conv_dims,
        matmul_size=conv_size,
        dispatch_kind=common.DispatchKind.conv,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
    )
    assert pipeline_options_space.use_igemm_convolution == [True]
    assert conv_size.K == [4608]
    assert conv_dims.k == [4]


def test_adjust_problem_size_for_pipeline_with_igemm_details(
    tuner_ctx: common.TunerContext,
) -> None:
    """Test adjust_problem_size_for_pipeline with IGEMM details from the binding."""
    context = tuner_ctx.mlir_ctx

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_hwcf(
            module=module,
            input_shape=(2, 32, 32, 128),
            kernel_shape=(3, 3, 128, 256),
            output_shape=(2, 30, 30, 256),
            input_type=tuner_ctx.type.f16,
            kernel_type=tuner_ctx.type.f16,
            output_type=tuner_ctx.type.f32,
        )

        root_op_list = iree_codegen.get_tuner_root_ops(module)
        assert len(root_op_list) == 1
        root_op = root_op_list[0]

        parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op, tuner_ctx)
        conv_op_info = parser.get_op_info()
        assert isinstance(conv_op_info, dispatch_parser.ConvolutionOpInfo)
        assert (
            conv_op_info.igemm_details is not None
        ), "IGEMM details should be available for NHWC conv"

        assert conv_op_info.dims.m == [0, 1, 2]
        assert conv_op_info.dims.n == [3]
        assert conv_op_info.dims.k == [4, 5, 6]
        assert conv_op_info.dims.batch == []

        assert conv_op_info.matmul_size.M == [2, 30, 30]
        assert conv_op_info.matmul_size.N == [256]
        assert conv_op_info.matmul_size.K == [3, 3, 128]
        assert conv_op_info.matmul_size.B == []

        conv_dims = common.ContractionDimensions(
            m=list(conv_op_info.dims.m),
            n=list(conv_op_info.dims.n),
            k=list(conv_op_info.dims.k),
            batch=list(conv_op_info.dims.batch),
        )
        conv_size = common.ContractionSizes(
            M=list(conv_op_info.matmul_size.M),
            N=list(conv_op_info.matmul_size.N),
            K=list(conv_op_info.matmul_size.K),
            B=list(conv_op_info.matmul_size.B),
        )

        pipeline_options_space = dispatch_constraints.PipelineOptionsSearchSpace(
            use_igemm_convolution=[None],
        )

        constraint_generator.adjust_problem_size_for_pipeline(
            contraction_dims=conv_dims,
            matmul_size=conv_size,
            dispatch_kind=common.DispatchKind.conv,
            pipeline_options_search_space=pipeline_options_space,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            igemm_details=conv_op_info.igemm_details,
        )

        # Verify that use_igemm_convolution is set.
        assert pipeline_options_space.use_igemm_convolution == [True]

        # Get expected IGEMM dimensions and bounds.
        igemm_maps = [
            map_attr.value
            for map_attr in conv_op_info.igemm_details.igemm_contraction_maps
        ]
        igemm_contraction_dims = linalg.infer_contraction_dimensions_from_maps(
            igemm_maps
        )
        igemm_bounds = list(conv_op_info.igemm_details.igemm_loop_bounds)

        # Verify that dimensions are updated to match IGEMM structure.
        assert conv_dims.m == list(igemm_contraction_dims.m)
        assert conv_dims.n == list(igemm_contraction_dims.n)
        assert conv_dims.k == list(igemm_contraction_dims.k)
        assert conv_dims.batch == list(igemm_contraction_dims.batch)

        # Verify sizes correspond to IGEMM loop bounds.
        expected_m_sizes = [igemm_bounds[i] for i in igemm_contraction_dims.m]
        expected_n_sizes = [igemm_bounds[i] for i in igemm_contraction_dims.n]
        expected_k_sizes = [igemm_bounds[i] for i in igemm_contraction_dims.k]

        assert conv_size.M == expected_m_sizes
        assert conv_size.N == expected_n_sizes
        assert conv_size.K == expected_k_sizes

        # Verify that K is flattened (3*3*128 = 1152).
        assert conv_size.K == [1152]
