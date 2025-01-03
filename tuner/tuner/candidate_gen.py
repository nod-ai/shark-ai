# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

"""
Generate candidates by tweaking op configuration for tuning.

It can be invoked in two ways:
    1. From another python script, import and call `tune()`
    2. Run this script directly from the command

Usage: ./candidate_gen.py 121.mlir -o "tuning/candidates" -l 1024 --lhs-dims=mk --rhs-dims=nk --tile-dims=mnk

"""

import argparse
import logging
import pickle
import re
from dataclasses import dataclass
from os import path, makedirs
from typing import Optional
from textwrap import indent
from abc import abstractmethod

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_codegen  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *
from .spec_builder import *

tune_logger = logging.getLogger("tune")


def apply_configuration(
    template: list[str],
    compilation_info: iree_codegen.CompilationInfoAttr,
) -> str:
    lowering_config = compilation_info.lowering_config
    intrinsic = lowering_config.mma_kind
    (
        subgroup_m_count,
        subgroup_n_count,
    ) = lowering_config.subgroup_count_mn
    workgroup_sizes = lowering_config.workgroup_tile_sizes
    reduction_sizes = lowering_config.reduction_tile_sizes
    gpu_pipeline_options = compilation_info.translation_info.configuration[
        GPU_PIPELINE_OPTIONS_KEY
    ]
    waves_per_eu = compilation_info.translation_info.configuration[LLVM_FUNC_ATTRS_KEY][
        WAVES_PER_EU_KEY
    ]
    tune_logger.info(f"Applying: {compilation_info}")
    expr0 = re.compile(
        r"<intrinsic = #iree_gpu\.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
    )
    expr1 = re.compile(
        r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
    )
    expr2 = re.compile(r"workgroup = \[([0-9]+)(, ([0-9]+))+\]")
    expr3 = re.compile(r"reduction = \[([0-9]+)(, ([0-9]+))+\]")
    expr4 = re.compile(r"gpu_pipeline_options = #iree_gpu\.pipeline_options<([^>]*)>")
    expr5 = re.compile(r"\"amdgpu-waves-per-eu\" = \"([0-9])\"")
    repl0 = f"<intrinsic = {intrinsic}, subgroup_m_count = {subgroup_m_count}, subgroup_n_count = {subgroup_n_count}>"
    repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, compilation_info.translation_info.workgroup_size))}] subgroup_size = {compilation_info.translation_info.subgroup_size},'
    repl2 = f"workgroup = {workgroup_sizes}"
    repl3 = f"reduction = {reduction_sizes}"
    repl4 = f"gpu_pipeline_options = {gpu_pipeline_options}"
    repl5 = f'"amdgpu-waves-per-eu" = {waves_per_eu}'

    new_mlir = ""
    for line in template:
        if "intrinsic =" in line:
            line = re.sub(expr0, repl0, line)
        if "LLVMGPUVectorDistribute " in line:
            line = re.sub(expr1, repl1, line)
        if "workgroup" in line:
            line = re.sub(expr2, repl2, line)
        if "reduction" in line:
            line = re.sub(expr3, repl3, line)
        if "gpu_pipeline_options =" in line:
            line = re.sub(expr4, repl4, line)
        if "amdgpu-waves-per-eu" in line:
            line = re.sub(expr5, repl5, line)
        new_mlir += line

    return new_mlir


class DispatchTuner(DispatchParser):
    # TODO(https://github.com/nod-ai/shark-ai/issues/453): Remove this in favor of configuring using transform dialect.
    @abstractmethod
    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        """Apply parameter transformations to the operation."""
        pass

    @abstractmethod
    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        """Generate a transform dialect spec that applies the compilation info attr."""
        pass


class DispatchTunerRegistry:
    def __init__(self, check_translation_info=True):
        self.check_translation_info = check_translation_info
        self.registry = set()

    def register(self, dispatch_tuners: list[DispatchTuner]) -> None:
        for dispatch_tuner in dispatch_tuners:
            self.registry.add(dispatch_tuner)

    # TODO(Max191): Remove translation info validation.
    def validate_translation(self, attrs: list[ir.NamedAttribute]) -> bool:
        if not self.check_translation_info:
            return True
        for attr in attrs:
            if (attr.name == "translation_info") and (
                "LLVMGPUVectorDistribute" in str(attr.attr)
            ):
                return True
        assert False, "Translation info not supported"

    def find_handler(self, op_name: str) -> DispatchTuner:
        for dispatch_tuner in self.registry:
            if dispatch_tuner.supports(op_name):
                return dispatch_tuner
        assert False, "Dispatch kind not supported"


class ContractionOpInterfaceTuner(DispatchTuner, ContractionOpInterfaceParser):
    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        raise NotImplementedError

    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        contraction_op: ir.Operation = self.get_contraction_operation(ir_module)
        lhs_type = ir.ShapedType(contraction_op.operands[0].type)
        rhs_type = ir.ShapedType(contraction_op.operands[1].type)
        acc_type = ir.ShapedType(contraction_op.operands[2].type)
        M = acc_type.get_dim_size(0)
        N = acc_type.get_dim_size(1)
        K = lhs_type.get_dim_size(1)
        # TODO(Max191): Get the function name from the func.func in the input module.
        func_name = f"match_contraction_{M}x{N}x{K}_{lhs_type.element_type}x{rhs_type.element_type}x{acc_type.element_type}"
        return build_td_spec(
            ir_module.context, contraction_op, compilation_info, func_name
        )


class ConvolutionOpInterfaceTuner(DispatchTuner, ConvolutionOpInterfaceParser):
    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        raise NotImplementedError

    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        conv_op: ir.Operation = self.get_conv_operation(ir_module)
        assert (
            conv_op.name == "linalg.conv_2d_nhwc_hwcf"
        ), "expected linalg.conv_2d_nhwc_hwcf"
        lhs_type = ir.ShapedType(conv_op.operands[0].type)
        rhs_type = ir.ShapedType(conv_op.operands[1].type)
        acc_type = ir.ShapedType(conv_op.operands[2].type)
        N = acc_type.get_dim_size(0)
        H = acc_type.get_dim_size(1)
        W = acc_type.get_dim_size(2)
        C = rhs_type.get_dim_size(2)
        P = rhs_type.get_dim_size(0)
        Q = rhs_type.get_dim_size(1)
        F = rhs_type.get_dim_size(3)
        conv_type = conv_op.name.split(".")[-1]
        # TODO(Max191): Get the function name from the func.func in the input module.
        func_name = f"match_{conv_type}_{N}x{H}x{W}x{C}x{P}x{Q}x{F}_{lhs_type.element_type}x{rhs_type.element_type}x{acc_type.element_type}"
        return build_td_spec(ir_module.context, conv_op, compilation_info, func_name)


class MmtTuner(DispatchTuner, MmtParser):
    def get_transform_function_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> str:
        return f"""
    transform.named_sequence @{functionName}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
    %config = transform.param.constant {compilation_info} -> !transform.any_param
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_mmt(
                problem_size, f"match_mmt_{M}x{N}x{K}", compilation_info
            ),
            "//   ",
        )
        modified += apply_configuration(
            template,
            compilation_info,
        )
        embeddable = indent(
            self.get_transform_function_mmt(
                problem_size, f"match_op", compilation_info
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        raise NotImplementedError


class ConvTuner(DispatchTuner, ConvParser):
    def get_transform_function_conv(
        self,
        problem_size: ProblemSize,
        functionName: str,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> str:
        dynamic_batch_input_ty = problem_size.lhs_type
        dynamic_batch_input_ty.shape = dynamic_batch_input_ty.shape.copy()
        dynamic_batch_input_ty.shape[0] = -1

        dynamic_batch_output_ty = problem_size.res_type
        dynamic_batch_output_ty.shape = dynamic_batch_output_ty.shape.copy()
        dynamic_batch_output_ty.shape[0] - 1

        input = f"tensor<{dynamic_batch_input_ty}>"
        filter = f"tensor<{problem_size.rhs_type}>"
        output = f"tensor<{dynamic_batch_output_ty}>"

        return f"""
    transform.named_sequence @{functionName}(%conv: !transform.any_op {{transform.readonly}})
    -> (!transform.any_op, !transform.any_param) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {{
    ^bb0(%lhs: {input}, %rhs: {filter}, %out: {output}):
        %13 = linalg.conv_2d_nhwc_hwcf {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}
        ins(%lhs, %rhs : {input}, {filter})
        outs(%out : {output}) -> {output}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
        %config = transform.param.constant {compilation_info} -> !transform.any_param
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        conv_dims = ConvDimInfo.from_problem_size(problem_size)
        modified = indent(
            self.get_transform_function_conv(
                problem_size,
                f"match_conv_2d_nhwc_hwcf_Bx{conv_dims.oh}x{conv_dims.ow}x{conv_dims.oc}x{conv_dims.fh}x{conv_dims.fw}x{conv_dims.ic}",
                compilation_info,
            ),
            "//   ",
        )
        modified += apply_configuration(
            template,
            compilation_info,
        )
        embeddable = indent(
            self.get_transform_function_conv(
                problem_size, f"match_op", compilation_info
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        raise NotImplementedError


class ContractionTuner(DispatchTuner, ContractionParser):
    def get_transform_function_broadcast_rhs_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> str:
        lhs_dynamic_batch = problem_size.lhs_type
        lhs_dynamic_batch.shape = lhs_dynamic_batch.shape.copy()
        lhs_dynamic_batch.shape[0] = -1

        return f"""
transform.named_sequence @{functionName}(%generic: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
%mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
%lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
%rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
transform.iree.match.cast_compatible_type %lhs = tensor<{lhs_dynamic_batch}> : !transform.any_value
transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
%config = transform.param.constant {compilation_info} -> !transform.any_param
transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""

    def apply_params_broadcast_rhs_mmt(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_broadcast_rhs_mmt(
                problem_size, f"match_broadcast_rhs_mmt_Bx{M}x{N}x{K}", compilation_info
            ),
            "//   ",
        )
        modified += apply_configuration(
            template,
            compilation_info,
        )

        embeddable = indent(
            self.get_transform_function_broadcast_rhs_mmt(
                problem_size, f"match_op", compilation_info
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        if self.is_broadcast_rhs_mmt(template):
            return self.apply_params_broadcast_rhs_mmt(
                problem_size, template, compilation_info
            )

        # TODO: Generate transform function.
        return MLIRTransformation(
            template,
            apply_configuration(
                template,
                compilation_info,
            ),
            "",
        )

    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        raise NotImplementedError


class BatchMmtTuner(DispatchTuner, BatchMmtParser):
    def get_transform_function_batch_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> str:
        return f"""
transform.named_sequence @{functionName}(%generic: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
%mmt = transform.include @match_batch_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
%lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
%rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
%config = transform.param.constant {compilation_info} -> !transform.any_param
transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        B = problem_size.matmul_size.B
        modified = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_batch_mmt_{B}x{M}x{N}x{K}", compilation_info
            ),
            "//   ",
        )
        modified += apply_configuration(
            template,
            compilation_info,
        )

        embeddable = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_op", compilation_info
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        raise NotImplementedError


class BatchMatmulTuner(DispatchTuner, BatchMatmulParser):
    def get_transform_function_batch_matmul(
        self,
        problem_size: ProblemSize,
        tile_dims: str,
        functionName: str,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> str:
        input0 = f"tensor<{problem_size.lhs_type}>"
        input1 = f"tensor<{problem_size.rhs_type}>"
        output = f"tensor<{problem_size.res_type}>"

        return f"""
    transform.named_sequence @{functionName}(%batch_matmul: !transform.any_op {{transform.readonly}})
    -> (!transform.any_op, !transform.any_param) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {{
    ^bb0(%lhs: {input0}, %rhs: {input1}, %out: {output}):
        %13 = linalg.batch_matmul
        ins(%lhs, %rhs : {input0}, {input1})
        outs(%out : {output}) -> {output}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
        %config = transform.param.constant {compilation_info} -> !transform.any_param
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_batch_matmul(
                problem_size,
                self.tile_dims,
                f"match_batch_matmul_{problem_size.matmul_size.B}x{M}x{N}x{K}",
                compilation_info,
            ),
            "//   ",
        )
        modified += apply_configuration(
            template,
            compilation_info,
        )

        embeddable = indent(
            self.get_transform_function_batch_matmul(
                problem_size, self.tile_dims, f"match_op", compilation_info
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def get_td_spec(
        self,
        ir_module: ir.Module,
        compilation_info: iree_codegen.CompilationInfoAttr,
    ) -> ir.Module:
        raise NotImplementedError


@dataclass
class OpWalkResult:
    was_interrupted: bool = False
    dispatch_tuner: Optional[DispatchTuner] = None


def walk_callback_get_fn(
    op: ir.Operation,
    walk_result: OpWalkResult,
    dispatch_tuner_registry: DispatchTunerRegistry,
) -> ir.WalkResult:
    if op.name == "func.func":
        dispatch_tuner_registry.validate_translation([a for a in op.opview.attributes])
    if op.name == "util.func":
        func_name = str(op.opview.sym_name)
        walk_result.was_interrupted = True
        walk_result.dispatch_tuner = dispatch_tuner_registry.find_handler(func_name)
        return ir.WalkResult.INTERRUPT
    return ir.WalkResult.ADVANCE


def walk_mlir_op(
    mlir_module: ir.Module,
    dispatch_tuner_registry: DispatchTunerRegistry,
) -> OpWalkResult:
    walk_result = OpWalkResult()
    for op in mlir_module.body.operations:
        op.walk(
            lambda op: walk_callback_get_fn(op, walk_result, dispatch_tuner_registry),
            ir.WalkOrder.POST_ORDER,
        )
        if walk_result.was_interrupted:
            break
    return walk_result


def get_default_output_dir() -> str:
    from datetime import datetime

    return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")


# TODO(https://github.com/nod-ai/shark-ai/issues/453): Remove in favor of using tune_with_td.
def tune(
    input: str,  # Path to the mlir file to be tuned
    output: str = "",  # Path to the output directory, auto creates one if not given
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints
    lhs_dims: str = "mk",  # Dimensions for the left-hand side operand in matrix operations
    rhs_dims: str = "nk",  # Dimensions for the right-hand side operand in matrix operations
    tile_dims: str = "mnk",  # Dimensions for the tile size
):
    input_file = str(input)

    if not output:
        output = get_default_output_dir()

    # Create the directory if it does not exist
    makedirs(str(output), exist_ok=True)

    tune_logger.debug(f"Output directory {output}")
    tune_logger.debug(f"Processing {input_file}")
    mlir_template = read_input_mlir(input_file)
    mlir_text = "".join(mlir_template)

    with ir.Context() as ctx:
        tuner_context = TunerContext(ctx, tune_logger)
        mlir_module = parse_mlir(mlir_text, tuner_context)
        # Save the input file as the first candidate.
        with open(path.join(output, f"0.mlir"), "w") as f:
            f.write(mlir_text)

        dispatch_tuner_registry = DispatchTunerRegistry()
        dispatch_tuner_registry.register(
            [
                MmtTuner(),
                ConvTuner(),
                ContractionTuner(lhs_dims, rhs_dims, tile_dims),
                BatchMmtTuner(),
                BatchMatmulTuner(lhs_dims, rhs_dims, tile_dims),
            ]
        )

        walk_result: OpWalkResult = walk_mlir_op(mlir_module, dispatch_tuner_registry)

        variant_op_list = iree_codegen.get_executable_variant_ops(mlir_module)
        assert len(variant_op_list) == 1, "Expect one executable variant op"
        variant_op = variant_op_list[0]
        # Get the MMA intrinisic intructions supported by the target.
        mma_list = iree_codegen.query_mma_intrinsics(variant_op)

        dispatch_tuner = walk_result.dispatch_tuner
        assert dispatch_tuner, "No suitable dispatch tuner found"
        problem_size: ProblemSize = dispatch_tuner.get_shapes(mlir_template)
        tune_logger.debug(str(problem_size))
        configs = []
        for i, config in enumerate(
            generate_solutions(tuner_context, problem_size, num_subgroups, mma_list)
        ):
            if i >= limit:
                break
            tune_logger.info(f"Solution #{i+1}: {config}")
            configs.append(config)
            tf_mlir = dispatch_tuner.apply_params(problem_size, mlir_template, config)

            with open(path.join(output, f"{i+1}.mlir"), "w") as f:
                f.write(tf_mlir.modified)
            with open(path.join(output, f"{i+1}_config.mlir"), "w") as f:
                f.write(tf_mlir.embeddable)

        # TODO: Fix pickling for ir types.
        # with open(path.join(output, "configs.pkl"), "wb") as file:
        #    pickle.dump(configs, file)

        tune_logger.info(f"Generated {len(configs)} candidates")
        tune_logger.info(f"Configurations .pkl is stored in {output}/configs.pkl")


def generate_configs_and_td_specs(
    input_module: ir.Module,  # Path to the mlir file to be tuned
    tuner_context: TunerContext,
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
) -> list[ir.Module]:
    dispatch_tuner_registry = DispatchTunerRegistry(check_translation_info=False)
    dispatch_tuner_registry.register(
        [
            ContractionOpInterfaceTuner(),
            ConvolutionOpInterfaceTuner(),
        ]
    )

    walk_result: OpWalkResult = walk_mlir_op(input_module, dispatch_tuner_registry)

    dispatch_tuner = walk_result.dispatch_tuner
    assert dispatch_tuner, "No suitable dispatch tuner found"
    problem_size: ProblemSize = dispatch_tuner.get_shapes(
        str(input_module).splitlines()
    )
    tune_logger.debug(str(problem_size))

    # Index 0 is reserved for default config, so it gets a placeholder spec.
    config_specs: list[ir.Module] = [get_placeholder_spec(input_module.context)]

    # Get the MMA intrinisic intructions supported by the target.
    variant_op_list = iree_codegen.get_executable_variant_ops(input_module)
    assert len(variant_op_list) == 1, "Expect one executable variant op"
    variant_op = variant_op_list[0]
    mma_list = iree_codegen.query_mma_intrinsics(variant_op)
    for i, config in enumerate(
        generate_solutions(
            tuner_context, problem_size, num_subgroups, mma_list, codegen_pipeline
        )
    ):
        if i >= limit:
            break
        tune_logger.info(f"Solution #{i+1}: {config}")
        td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
        assert td_spec_module, "Failed to generate transform dialect spec"
        config_specs.append(td_spec_module)

    tune_logger.info(f"Generated {len(config_specs)} tuning specs")
    return config_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input mlir file", type=str)
    parser.add_argument(
        "-o", "--output", help="Output dir", type=str, default=get_default_output_dir()
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="Max number of candidates generated",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--num-subgroups",
        help="Number of subgroups per workgroup to use. (-1 == unconstrained)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--lhs-dims", help="Map of LHS matmul dims", type=str, default="mk"
    )
    parser.add_argument(
        "--rhs-dims", help="Map of RHS matmul dims", type=str, default="nk"
    )
    parser.add_argument(
        "--tile-dims", help="Map of tile size matmul dims", type=str, default="mnk"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
    )

    args = parser.parse_args()
    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Create printing formatter for logging info
    formatter = logging.Formatter("%(message)s")

    # Create a handler to print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    tune_logger.addHandler(console_handler)

    # # Optionally, add a file handler to log to a file
    # file_handler = logging.FileHandler("tune.log")
    # file_handler.setFormatter(formatter)
    # tune_logger.addHandler(file_handler)

    tune(
        args.input,
        args.output,
        args.limit,
        args.num_subgroups,
        args.lhs_dims,
        args.rhs_dims,
        args.tile_dims,
    )


if __name__ == "__main__":
    args = main()
