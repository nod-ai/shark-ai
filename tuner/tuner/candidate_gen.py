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
    1. From another python script, import and call `generate_configs_and_td_specs()`
    2. Run this script directly from the command
Usage: python -m tuner.candidate_gen mmt_benchmark.mlir -o spec_dir -l 1024
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
import subprocess
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


@dataclass
class RunPack:
    command: list[str]
    check: bool = True
    timeout_seconds: Optional[int] = None


@dataclass
class RunResult:
    process_res: Optional[subprocess.CompletedProcess]
    is_timeout: bool


def run_command(run_pack: RunPack) -> RunResult:
    command = run_pack.command
    check = run_pack.check
    timeout_seconds = run_pack.timeout_seconds

    result = None
    is_timeout = False
    try:
        # Convert the command list to a command string for logging
        command_str = " ".join(command)
        logging.debug(f"Run: {command_str}")

        # Add timeout to subprocess.run call
        result = subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.stdout:
            logging.debug(f"stdout: {result.stdout}")
        if result.stderr:
            logging.debug(f"stderr: {result.stderr}")
    except subprocess.TimeoutExpired as e:
        logging.warning(
            f"Command '{command_str}' timed out after {timeout_seconds} seconds."
        )
        is_timeout = True
    except subprocess.CalledProcessError as e:
        print(e.output)
        logging.error(
            f"Command '{command_str}' returned non-zero exit status {e.returncode}."
        )
        logging.error(f"Command '{command_str}' failed with error: {e.stderr}")
        if check:
            raise
    except KeyboardInterrupt:
        print("Ctrl+C detected, terminating child processes...")

    return RunResult(result, is_timeout)


# The `strip_root_op_attr` and `strip_compilation_info` functions are used for
# getting consistent inputs to the compilation step in tuning. Inputs may come
# in with lowering configs, translation info, and root_op attrs when the input
# is a benchmark, but not when the input is a source MLIR file. Stripping the
# info makes the inputs to compilation consistent, and allows for overwriting
# the compilation info with generated TD specs during codegen.
def strip_root_op_attr(module: ir.Module):
    root_ops: list[ir.Operation] = get_ops_from_module(module, is_root_op)
    for root_op in root_ops:
        assert (
            ROOT_OP_ATTR_NAME in root_op.opview.attributes
        ), f"expected root op to have '{ROOT_OP_ATTR_NAME}' attr"
        del root_op.opview.attributes[ROOT_OP_ATTR_NAME]


# See the above comment for `strip_root_op_attr`.
def strip_compilation_info(input_path: Path) -> str:
    # Strip compilation info from the source and save the stripped IR
    strip_command = [
        f"iree-opt",
        f"{input_path}",
        f"--iree-codegen-strip-compilation-info",
    ]
    result = run_command(
        RunPack(
            command=strip_command,
            check=True,
        )
    )
    assert (
        result.process_res is not None
    ), "expected result from stripping compilation info"
    return result.process_res.stdout


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

    with ir.Context() as ctx:
        tuner_ctx = TunerContext(ctx, tune_logger)
        mlir_text = strip_compilation_info(args.input)
        mlir_module = parse_mlir(mlir_text, tuner_ctx)
        specs = generate_configs_and_td_specs(
            mlir_module,
            tuner_ctx,
            args.limit,
            args.num_subgroups,
            iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        )
        for candidate_num, spec in enumerate(specs):
            spec_dir = Path(args.output)
            spec_path = spec_dir / f"{candidate_num}_spec.mlir"
            spec_dir.mkdir(parents=True, exist_ok=True)
            with open(spec_path, "w") as f:
                f.write(str(spec))


if __name__ == "__main__":
    args = main()
