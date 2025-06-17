# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import z3  # type: ignore
import math
from abc import ABC, abstractmethod
from typing import Iterator

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore
from iree.compiler.dialects import linalg  # type: ignore

from . import common
from . import dispatch_constraints


def adjust_problem_size_for_pipeline(
    contraction_dims: common.ContractionDimensions,
    matmul_size: common.ContractionSizes,
    dispatch_kind: common.DispatchKind,
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
):
    # Adjustment is only needed for IGEMM. Fail if the problem is not a conv
    # going down the TileAndFuse pipeline.
    if (
        codegen_pipeline != iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
        or dispatch_kind != common.DispatchKind.conv
    ):
        return

    pipeline_options_search_space.use_igemm_convolution = [True]

    # Flatten the K dimensions into a single dimension for IGEMM lowering.
    contraction_dims.k = [contraction_dims.k[0]]
    matmul_size.K = [math.prod(matmul_size.K)]


def generate_generic_contraction_solutions(
    tuner_ctx: common.TunerContext,
    contraction_dims: common.ContractionDimensions,
    matmul_size: common.ContractionSizes,
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    dispatch_kind: common.DispatchKind,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    num_subgroups: int = 4,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic] = [],
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
) -> Iterator[list[common.TuningConfiguration]]:
    adjust_problem_size_for_pipeline(
        contraction_dims,
        matmul_size,
        dispatch_kind,
        pipeline_options_search_space,
        codegen_pipeline,
    )

    M, N, K = matmul_size.M, matmul_size.N, matmul_size.K
    tuner_ctx.logger.debug(f"{M},{N},{K}")

    m_vars = [z3.Int(f"m{i}") for i in range(len(M))]
    n_vars = [z3.Int(f"n{i}") for i in range(len(N))]
    k_vars = [z3.Int(f"k{i}") for i in range(len(K))]
    subgroup_m_vars = [z3.Int(f"subgroup_m{i}") for i in range(len(M))]
    subgroup_n_vars = [z3.Int(f"subgroup_n{i}") for i in range(len(N))]

    subgroup_size = z3.Int("subgroup_size")
    intrinsic_mn = z3.Int("intrinsic_mn")
    intrinsic_k = z3.Int("intrinsic_k")
    wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")
    all_vars = (
        m_vars
        + n_vars
        + k_vars
        + [
            subgroup_size,
            intrinsic_mn,
            intrinsic_k,
            wg_x,
            wg_y,
            wg_z,
            sg_m_cnt,
            sg_n_cnt,
        ]
    )

    solver = z3.Solver()
    match codegen_pipeline:
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute:
            constraints = dispatch_constraints.generate_vector_distribute_constraints(
                matmul_size,
                lhs_type,
                rhs_type,
                res_type,
                [m_vars, n_vars, k_vars],
                num_subgroups,
                subgroup_size,
                [intrinsic_mn, intrinsic_k],
                [wg_x, wg_y, wg_z],
                sg_m_cnt,
                sg_n_cnt,
                mma_intrinsics,
                dispatch_kind,
            )
            constraints += [v == 0 for v in subgroup_m_vars + subgroup_n_vars]
        case iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse:
            constraints = dispatch_constraints.generate_tile_and_fuse_constraints(
                matmul_size,
                lhs_type,
                rhs_type,
                res_type,
                [m_vars, n_vars, k_vars, subgroup_m_vars, subgroup_n_vars],
                num_subgroups,
                subgroup_size,
                [intrinsic_mn, intrinsic_k],
                [wg_x, wg_y, wg_z],
                sg_m_cnt,
                sg_n_cnt,
                mma_intrinsics,
            )

    solver.add(z3.simplify(z3.And(constraints)))
    tuner_ctx.logger.debug(f"Initial constraints: {solver}")

    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()
        intrinsic_mnk_shape = (
            lookup(intrinsic_mn),
            lookup(intrinsic_mn),
            lookup(intrinsic_k),
        )
        mma_attr = dispatch_constraints.getMMAAttr(
            res_type.element_type,
            *intrinsic_mnk_shape,
            lhs_type.element_type,
            rhs_type.element_type,
        )

        def set_cdim_tile_sizes(tile_sizes, contraction_dims, csizes):
            for dim, size in zip(contraction_dims, csizes):
                tile_sizes[dim] = size

        # Get workgroup tile sizes.
        workgroup_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            contraction_dims.m,
            [lookup(v) for v in m_vars],
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            contraction_dims.n,
            [lookup(v) for v in n_vars],
        )
        set_cdim_tile_sizes(
            workgroup_tile_sizes,
            contraction_dims.batch,
            [1] * len(contraction_dims.batch),
        )

        # Get subgroup tile sizes.
        subgroup_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            contraction_dims.m,
            [lookup(v) for v in subgroup_m_vars],
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            contraction_dims.n,
            [lookup(v) for v in subgroup_n_vars],
        )
        set_cdim_tile_sizes(
            subgroup_tile_sizes,
            contraction_dims.batch,
            [1] * len(contraction_dims.batch),
        )

        # Get reduction tile sizes.
        reduction_tile_sizes = [0] * (
            len(M) + len(N) + len(K) + len(contraction_dims.batch)
        )
        set_cdim_tile_sizes(
            reduction_tile_sizes,
            contraction_dims.k,
            [lookup(v) for v in k_vars],
        )

        required_padding = any(
            p[-1] % i != 0 for p, i in zip((M, N, K), intrinsic_mnk_shape, strict=True)
        )
        promote_operands = [0, 1]
        padding = None
        if required_padding:
            # TODO: Remove promotion of operand 2 once codegen supports handling padded outputs without promotion.
            promote_operands = [0, 1, 2]
            _, _, mma_intrinsic_k = mma_attr.mnk_shape
            padding = [
                *(workgroup_tile_sizes[d] for d in contraction_dims.m),
                *(workgroup_tile_sizes[d] for d in contraction_dims.n),
                *(
                    reduction_tile_sizes[d] * mma_intrinsic_k
                    for d in contraction_dims.k
                ),
            ]

        compilation_infos = dispatch_constraints.generate_compilation_infos(
            tuner_ctx,
            mma_attr,
            workgroup_tile_sizes,
            reduction_tile_sizes,
            subgroup_tile_sizes,
            (lookup(wg_x), lookup(wg_y), lookup(wg_z)),
            lookup(subgroup_size),
            lookup(sg_m_cnt),
            lookup(sg_n_cnt),
            promote_operands,
            codegen_pipeline,
            pipeline_options_search_space,
            allowed_waves_per_eu,
            padding=padding,
        )

        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1

        for compilation_info in compilation_infos:
            yield [
                common.TuningConfiguration(
                    name="compilation_info", configuration=compilation_info
                )
            ]


class ConstraintGenerator(ABC):
    """
    Describes how to generate constraints and produce tuning candidates
    for a specific type of tunable problem.

    Implementations of ConstraintGenerator are responsible for encapsulating
    problem-specific information—such as contraction dimensions, sizes, operand types—
    and using that information to generate valid configurations that satisfy the
    constraints imposed by the codegen pipeline and target architecture.

    The `generate_solutions` method returns an iterator over lists of
    `TuningConfiguration` instances. Each list represents a self-contained tuning
    candidate that can be applied to the dispatch root op.

    Example output:
        [
            TuningConfiguration(name="compilation_info", configuration=CompilationInfoAttr(...)),
            TuningConfiguration(name="decomposition_config", configuration=DecompositionConfigAttr(...)),
        ]
    """

    @abstractmethod
    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        """
        Generate a sequence of tuning configuration entries for the specified pipeline.
        """
        pass


class ContractionOpInterfaceConstraintGenerator(ConstraintGenerator):
    def __init__(self, root_op: ir.Operation):
        self.root_op = root_op
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        assert contraction_dims, "no contraction dimensions"
        dims = common.ContractionDimensions(
            batch=list(contraction_dims.batch),
            m=list(contraction_dims.m),
            n=list(contraction_dims.n),
            k=list(contraction_dims.k),
        )

        res_maps = linalg.get_indexing_maps(root_op)
        maps = [map_attr.value for map_attr in res_maps]
        lhs_dims = common.get_map_result_dim_positions(maps[0])
        rhs_dims = common.get_map_result_dim_positions(maps[1])
        res_dims = common.get_map_result_dim_positions(maps[2])

        assert lhs_dims, "no lhs dimensions"
        assert rhs_dims, "no rhs dimensions"
        assert res_dims, "no result dimensions"

        lhs_type = ir.RankedTensorType(root_op.operands[0].type)
        rhs_type = ir.RankedTensorType(root_op.operands[1].type)
        res_type = ir.RankedTensorType(root_op.operands[2].type)

        matmul_size = common.ContractionSizes(
            M=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.m],
            N=[rhs_type.shape[rhs_dims.index(dim)] for dim in contraction_dims.n],
            K=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.k],
            B=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch],
        )

        self.dims = dims
        self.matmul_size = matmul_size
        self.lhs_type = common.ShapedType(lhs_type.shape, lhs_type.element_type)
        self.rhs_type = common.ShapedType(rhs_type.shape, rhs_type.element_type)
        self.res_type = common.ShapedType(res_type.shape, res_type.element_type)

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            contraction_dims=self.dims,
            matmul_size=self.matmul_size,
            lhs_type=self.lhs_type,
            rhs_type=self.rhs_type,
            res_type=self.res_type,
            dispatch_kind=common.DispatchKind.contraction,
            codegen_pipeline=codegen_pipeline,
            **pipeline_constraint_options,
        )


class ConvolutionOpInterfaceConstraintGenerator(ConstraintGenerator):
    def __init__(self, root_op: ir.Operation):
        self.root_op = root_op

        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        assert convolution_dims, "no convolution dimensions"
        contraction_dims = common.ContractionDimensions(
            batch=list(convolution_dims.depth),
            m=list(convolution_dims.batch) + list(convolution_dims.output_image),
            n=list(convolution_dims.output_channel),
            k=list(convolution_dims.filter_loop) + list(convolution_dims.input_channel),
        )

        def find_iter_dim_size(iter_dim: int, operand: int):
            operand_type = root_op.operands[operand].type
            indexing_map = linalg.get_indexing_maps(root_op)[operand]
            tensor_dim = list(indexing_map.value.results).index(
                ir.AffineExpr.get_dim(iter_dim)
            )
            return operand_type.shape[tensor_dim]

        matmul_size = common.ContractionSizes(
            B=[find_iter_dim_size(d, operand=2) for d in contraction_dims.batch],
            M=[find_iter_dim_size(d, operand=2) for d in contraction_dims.m],
            N=[find_iter_dim_size(d, operand=2) for d in contraction_dims.n],
            K=[find_iter_dim_size(d, operand=1) for d in contraction_dims.k],
        )

        lhs_type = root_op.operands[0].type
        rhs_type = root_op.operands[1].type
        res_type = root_op.operands[2].type

        self.dims = contraction_dims
        self.matmul_size = matmul_size
        self.lhs_type = common.ShapedType(lhs_type.shape, lhs_type.element_type)
        self.rhs_type = common.ShapedType(rhs_type.shape, rhs_type.element_type)
        self.res_type = common.ShapedType(res_type.shape, res_type.element_type)

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            contraction_dims=self.dims,
            matmul_size=self.matmul_size,
            lhs_type=self.lhs_type,
            rhs_type=self.rhs_type,
            res_type=self.res_type,
            dispatch_kind=common.DispatchKind.conv,
            codegen_pipeline=codegen_pipeline,
            **pipeline_constraint_options,
        )
