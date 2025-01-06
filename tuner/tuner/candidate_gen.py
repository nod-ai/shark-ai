# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

"""
Generate candidates by tweaking op configuration for tuning.
"""

# import argparse
import logging
from dataclasses import dataclass
from typing import Optional
from abc import abstractmethod

from iree.compiler import ir  # type: ignore

from iree.compiler.dialects import iree_codegen  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *
from .spec_builder import *

tune_logger = logging.getLogger("tune")


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


def generate_configs_and_td_specs(
    input_module: ir.Module,  # Path to the mlir file to be tuned
    tuner_context: TunerContext,
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints
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
        generate_solutions(tuner_context, problem_size, num_subgroups, mma_list)
    ):
        if i >= limit:
            break
        tune_logger.info(f"Solution #{i+1}: {config}")
        td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
        assert td_spec_module, "Failed to generate transform dialect spec"
        config_specs.append(td_spec_module)

    tune_logger.info(f"Generated {len(config_specs)} tuning specs")
    return config_specs
