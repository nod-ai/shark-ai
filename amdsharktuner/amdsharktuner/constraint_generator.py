# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import z3  # type: ignore
import math
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from iree.compiler.dialects import iree_codegen, iree_gpu, linalg  # type: ignore

from . import common, dispatch_constraints, dispatch_parser


def adjust_problem_size_for_pipeline(
    contraction_dims: common.ContractionDimensions,
    matmul_size: common.ContractionSizes,
    dispatch_kind: common.DispatchKind,
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
    igemm_details: Optional[iree_codegen.IGEMMGenericConvDetails] = None,
):
    # Adjustment is only needed for IGEMM. Fail if the problem is not a conv
    # going down the TileAndFuse pipeline.
    if (
        codegen_pipeline != iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
        or dispatch_kind != common.DispatchKind.conv
    ):
        return

    pipeline_options_search_space.use_igemm_convolution = [True]

    # Use IGEMM binding details if available for accurate dimension mapping.
    if igemm_details:
        igemm_maps = [
            map_attr.value for map_attr in igemm_details.igemm_contraction_maps
        ]
        igemm_contraction_dims = linalg.infer_contraction_dimensions_from_maps(
            igemm_maps
        )
        assert (
            igemm_contraction_dims
        ), "Failed to infer contraction dimensions from IGEMM maps"

        bounds = list(igemm_details.igemm_loop_bounds)

        # Update contraction_dims with IGEMM structure.
        contraction_dims.m = list(igemm_contraction_dims.m)
        contraction_dims.n = list(igemm_contraction_dims.n)
        contraction_dims.k = list(igemm_contraction_dims.k)
        contraction_dims.batch = list(igemm_contraction_dims.batch)

        # Update matmul_size with IGEMM loop bounds (K is already flattened!).
        matmul_size.M = [bounds[i] for i in contraction_dims.m]
        matmul_size.N = [bounds[i] for i in contraction_dims.n]
        matmul_size.K = [bounds[i] for i in contraction_dims.k]
        matmul_size.B = [bounds[i] for i in contraction_dims.batch]

        return

    # Fallback: Manual flattening for legacy path when IGEMM details are unavailable.
    # TODO(Bangtian): Once all IGEMM implementation is complete, fully remove this fallback path
    # and corresponding tests.
    contraction_dims.k = [contraction_dims.k[0]]
    matmul_size.K = [math.prod(matmul_size.K)]


def generate_generic_contraction_solutions(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    contraction_dims: common.ContractionDimensions,
    matmul_size: common.ContractionSizes,
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    dispatch_kind: common.DispatchKind,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    num_subgroups: int = 4,
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
    igemm_details: Optional[iree_codegen.IGEMMGenericConvDetails] = None,
) -> Iterator[list[common.TuningConfiguration]]:
    adjust_problem_size_for_pipeline(
        contraction_dims,
        matmul_size,
        dispatch_kind,
        pipeline_options_search_space,
        codegen_pipeline,
        igemm_details,
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
                gpu_target_info,
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
                gpu_target_info,
            )

    solver.add(z3.simplify(z3.And(constraints)))
    tuner_ctx.logger.debug(f"Initial constraints: {solver}")

    num_loops = (
        len(contraction_dims.m)
        + len(contraction_dims.n)
        + len(contraction_dims.k)
        + len(contraction_dims.batch)
    )

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
            gpu_target_info.mma_intrinsics,
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
        # Setting subgroup basis.
        # TODO(Bangtian): Sync changes from IREE PR: https://github.com/iree-org/iree/pull/22000.
        subgroup_basis_counts = [1] * num_loops
        m_dim = contraction_dims.m[-1]
        subgroup_basis_counts[m_dim] = lookup(sg_m_cnt)
        n_dim = contraction_dims.n[-1]
        subgroup_basis_counts[n_dim] = lookup(sg_n_cnt)
        subgroup_basis_mapping = list(range(num_loops))

        compilation_infos = dispatch_constraints.generate_compilation_infos(
            tuner_ctx,
            mma_attr,
            workgroup_tile_sizes,
            reduction_tile_sizes,
            subgroup_tile_sizes,
            (lookup(wg_x), lookup(wg_y), lookup(wg_z)),
            lookup(subgroup_size),
            subgroup_basis_counts,
            subgroup_basis_mapping,
            promote_operands,
            codegen_pipeline,
            pipeline_options_search_space,
            allowed_waves_per_eu,
            padding=padding,
        )

        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1
        knob_assignment = None
        for compilation_info in compilation_infos:
            if (
                codegen_pipeline
                == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
            ):
                knob_assignment = common.LLVMGPUVectorDistributeContractionKnobs(
                    M=int(math.prod(M)),
                    N=int(math.prod(N)),
                    K=int(math.prod(K)),
                    tile_m=workgroup_tile_sizes[0],
                    tile_n=workgroup_tile_sizes[1],
                    tile_k=reduction_tile_sizes[2],
                    wg_x=lookup(wg_x),
                    wg_y=lookup(wg_y),
                    wg_z=lookup(wg_z),
                    subgroup_m_cnt=lookup(sg_m_cnt),
                    subgroup_n_cnt=lookup(sg_n_cnt),
                    intrinsic_mn=lookup(intrinsic_mn),
                    intrinsic_k=lookup(intrinsic_k),
                    subgroup_m=subgroup_tile_sizes[0],
                    subgroup_n=subgroup_tile_sizes[1],
                    subgroup_k=subgroup_tile_sizes[2],
                )
            yield [
                common.TuningConfiguration(
                    name="compilation_info",
                    configuration=compilation_info,
                    knob_assignment=knob_assignment,
                )
            ]


def generate_attention_solutions(
    tuner_ctx: common.TunerContext,
    gpu_target_info: iree_gpu.TargetInfo,
    op_info: dispatch_parser.AttentionOpInfo,
    dispatch_kind: common.DispatchKind,
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    num_subgroups: int = 4,
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
) -> Iterator[list[common.TuningConfiguration]]:
    if (
        dispatch_kind != common.DispatchKind.attention
        or codegen_pipeline
        != iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    ):
        return []

    m_var = z3.Int("m_tile")
    n_var = z3.Int("n_tile")
    k_var = z3.Int("k_tile")

    subgroup_size = z3.Int("subgroup_size")
    qk_intrinsic_mn = z3.Int("qk_intrinsic_mn")
    qk_intrinsic_k = z3.Int("qk_intrinsic_k")
    pv_intrinsic_mn = z3.Int("pv_intrinsic_mn")
    pv_intrinsic_k = z3.Int("pv_intrinsic_k")
    sg_m_cnt = z3.Int("sg_m_cnt")
    sg_n_cnt = z3.Int("sg_n_cnt")

    # Used to determine if prefetch_shared_memory can be enabled.
    # See: https://github.com/iree-org/iree/blob/411aa64083a2303946b4d2d72d00e6a6814fbafb/compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp#L974-L976.
    can_reuse_qk_output_for_pv_input = z3.Bool("can_reuse_qk_output_for_pv_input")

    all_vars = (
        [m_var]
        + [n_var]
        + [k_var]
        + [
            subgroup_size,
            qk_intrinsic_mn,
            qk_intrinsic_k,
            pv_intrinsic_mn,
            pv_intrinsic_k,
            sg_m_cnt,
            sg_n_cnt,
        ]
    )

    solver = z3.Solver()
    constraints = dispatch_constraints.generate_attention_vector_distribute_constraints(
        op_info.qk_matmul,
        op_info.pv_matmul,
        op_info.transposed_q,
        op_info.transposed_k,
        op_info.transposed_v,
        [m_var, n_var, k_var],
        num_subgroups,
        subgroup_size,
        [qk_intrinsic_mn, qk_intrinsic_k],
        [pv_intrinsic_mn, pv_intrinsic_k],
        sg_m_cnt,
        sg_n_cnt,
        can_reuse_qk_output_for_pv_input,
        gpu_target_info,
    )

    solver.add(z3.simplify(z3.And(constraints)))
    tuner_ctx.logger.debug(f"Initial constraints: {solver}")

    i = 0
    while solver.check() == z3.sat:
        model = solver.model()
        lookup = lambda var: model[var].as_long()
        qk_intrinsic_mnk_shape = (
            lookup(qk_intrinsic_mn),
            lookup(qk_intrinsic_mn),
            lookup(qk_intrinsic_k),
        )
        qk_mma_attr = dispatch_constraints.getMMAAttr(
            op_info.qk_matmul.acc_type,
            *qk_intrinsic_mnk_shape,
            op_info.qk_matmul.lhs_type,
            op_info.qk_matmul.rhs_type,
            gpu_target_info.mma_intrinsics,
        )

        pv_intrinsic_mnk_shape = (
            lookup(pv_intrinsic_mn),
            lookup(pv_intrinsic_mn),
            lookup(pv_intrinsic_k),
        )
        pv_mma_attr = dispatch_constraints.getMMAAttr(
            op_info.pv_matmul.acc_type,
            *pv_intrinsic_mnk_shape,
            op_info.pv_matmul.lhs_type,
            op_info.pv_matmul.rhs_type,
            gpu_target_info.mma_intrinsics,
        )

        # Get workgroup tile sizes.
        workgroup_tile_sizes = [0] * op_info.domain_rank
        reduction_tile_sizes = [0] * op_info.domain_rank

        for b in op_info.batch_dims:
            workgroup_tile_sizes[b] = 1
        for m in op_info.m_dims[:-1]:
            workgroup_tile_sizes[m] = 1
        for n in op_info.n_dims[:-1]:
            workgroup_tile_sizes[n] = 1
        for k2 in op_info.k2_dims[:-1]:
            reduction_tile_sizes[k2] = 1

        workgroup_tile_sizes[op_info.m_dims[-1]] = lookup(m_var)
        workgroup_tile_sizes[op_info.n_dims[-1]] = lookup(n_var)
        reduction_tile_sizes[op_info.k2_dims[-1]] = lookup(k_var)

        subgroup_basis_counts = [1] * op_info.domain_rank
        subgroup_basis_mapping = list(range(op_info.domain_rank))
        subgroup_basis_counts[op_info.m_dims[-1]] = lookup(sg_m_cnt)
        subgroup_basis_counts[op_info.n_dims[-1]] = lookup(sg_n_cnt)
        qk_basis_mapping = [
            mapping
            for i, mapping in enumerate(subgroup_basis_mapping)
            if i not in op_info.n_dims
        ]
        qk_config = {
            "mma_kind": qk_mma_attr,
            "subgroup_basis": [subgroup_basis_counts, qk_basis_mapping],
            "promote_operands": [0, 1],
        }

        qk_lowering_config = common.get_lowering_config(
            tuner_ctx=tuner_ctx, **qk_config
        )

        pv_basis_mapping = [
            mapping
            for i, mapping in enumerate(subgroup_basis_mapping)
            if i not in op_info.k1_dims
        ]
        pv_config = {
            "mma_kind": pv_mma_attr,
            "subgroup_basis": [subgroup_basis_counts, pv_basis_mapping],
            "promote_operands": [1],
        }
        pv_lowering_config = common.get_lowering_config(
            tuner_ctx=tuner_ctx, **pv_config
        )

        decomposition_config = common.get_attention_decomposition_config(
            tuner_ctx, qk_lowering_config, pv_lowering_config
        )

        workgroup_size = lookup(sg_m_cnt) * lookup(sg_n_cnt) * lookup(subgroup_size)

        # Set prefetch_shared_memory based on whether layouts match.
        layouts_match = bool(model[can_reuse_qk_output_for_pv_input])
        pipeline_options_search_space.prefetch_shared_memory = [layouts_match]

        promote_operands = [0, 1, 2]
        compilation_infos = dispatch_constraints.generate_compilation_infos(
            tuner_ctx,
            None,
            workgroup_tile_sizes,
            reduction_tile_sizes,
            [0, 0, 0],
            (workgroup_size, 1, 1),
            lookup(subgroup_size),
            subgroup_basis_counts,
            subgroup_basis_mapping,
            promote_operands,
            codegen_pipeline,
            pipeline_options_search_space,
            allowed_waves_per_eu,
            padding=None,
        )
        solver.add(z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars)))))
        i += 1

        for compilation_info in compilation_infos:
            config_list = [
                common.TuningConfiguration(
                    name="compilation_info", configuration=compilation_info
                ),
                common.TuningConfiguration(
                    name="decomposition_config", configuration=decomposition_config
                ),
            ]
            yield config_list


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
        gpu_target_info: iree_gpu.TargetInfo,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        """
        Generate a sequence of tuning configuration entries for the specified pipeline.
        """
        pass


class ContractionOpInterfaceConstraintGenerator(ConstraintGenerator):
    def __init__(self, op_info: dispatch_parser.ContractionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.contraction,
            codegen_pipeline=codegen_pipeline,
            **pipeline_constraint_options,
        )


class ConvolutionOpInterfaceConstraintGenerator(ConstraintGenerator):
    def __init__(self, op_info: dispatch_parser.ConvolutionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.conv,
            codegen_pipeline=codegen_pipeline,
            igemm_details=self.op_info.igemm_details,
            **pipeline_constraint_options,
        )


class AttentionOpInterfaceConstraintGenerator(ConstraintGenerator):
    """
    Constraint generator for the IREE LinalgExt AttentionOp.

    Generates tuning configurations for attention operations.

    Attributes:
        op_info: AttentionOpInfo containing all attention operation metadata.
    """

    def __init__(self, op_info: dispatch_parser.AttentionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return generate_attention_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            op_info=self.op_info,
            dispatch_kind=common.DispatchKind.attention,
            codegen_pipeline=codegen_pipeline,
            **pipeline_constraint_options,
        )
