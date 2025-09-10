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
Usage: python -m sharktuner.candidate_gen mmt_benchmark.mlir -o spec_dir -l 1024
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional
from abc import abstractmethod

import iree.compiler as ireec  # type: ignore
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from . import (
    common,
    dispatch_constraints,
    dispatch_parser,
    spec_builder,
    constraint_generator,
)

tune_logger = logging.getLogger("tune")


class DispatchTuner(dispatch_parser.DispatchParser):
    @abstractmethod
    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        """
        Generates a transform dialect spec from a list of TuningConfiguration objects.

        Each TuningConfiguration specifies a name (e.g., "compilation_info") and
        its corresponding MLIR attribute (e.g., CompilationInfoAttr) to be applied
        to the dispatch root operation.
        """
        pass

    @abstractmethod
    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        """Returns a ConstraintGenerator associated with this dispatch root op."""
        pass


class DispatchTunerRegistry:
    def __init__(self):
        self.registry = set()

    def register(self, dispatch_tuners: list[DispatchTuner]) -> None:
        for dispatch_tuner in dispatch_tuners:
            self.registry.add(dispatch_tuner)

    def find_handler(self, op_name: str) -> DispatchTuner:
        for dispatch_tuner in self.registry:
            if dispatch_tuner.supports(op_name):
                return dispatch_tuner
        assert False, "Dispatch kind not supported"


class ContractionOpInterfaceTuner(
    DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return constraint_generator.ContractionOpInterfaceConstraintGenerator(
            self.get_root_op()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        contraction_op = self.get_root_op()
        func_name = self.get_root_op_func_name()
        return spec_builder.build_td_spec(
            contraction_op.context, contraction_op, config_list, func_name
        )


class ConvolutionOpInterfaceTuner(
    DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return constraint_generator.ConvolutionOpInterfaceConstraintGenerator(
            self.get_root_op()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        conv_op = self.get_root_op()
        func_name = self.get_root_op_func_name()
        return spec_builder.build_td_spec(
            conv_op.context, conv_op, config_list, func_name
        )


class AttentionOpInterfaceTuner(
    DispatchTuner, dispatch_parser.AttentionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation):
        super().__init__(root_op)

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return constraint_generator.AttentionOpInterfaceConstraintGenerator(
            self.get_root_op()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        attention_op = self.get_root_op()
        func_name = self.get_root_op_func_name()
        return spec_builder.build_td_spec(
            attention_op.context, attention_op, config_list, func_name
        )


def get_default_output_dir() -> str:
    from datetime import datetime

    return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")


def generate_configs_and_td_specs(
    input_module: ir.Module,  # Path to the mlir file to be tuned
    tuner_context: common.TunerContext,
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
) -> list[ir.Module]:
    dispatch_tuners: list[type[DispatchTuner]] = [
        ContractionOpInterfaceTuner,
        ConvolutionOpInterfaceTuner,
        AttentionOpInterfaceTuner,
    ]

    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
    if len(root_op_list) == 0:
        tune_logger.error(
            "No root ops found. Did you forget to pass "
            "--iree-config-add-tuner-attributes during compilation?"
        )
        return []
    elif len(root_op_list) > 1:
        tune_logger.error("Multiple root ops found. Only one is currently supported.")
        return []

    root_op = root_op_list[0]

    dispatch_tuner: Optional[DispatchTuner] = None
    for tuner_class in dispatch_tuners:
        tuner = tuner_class(root_op)
        if tuner.has_valid_root_op():
            dispatch_tuner = tuner
            break

    assert dispatch_tuner, "No suitable dispatch tuner found"

    # Index 0 is reserved for default config, so it gets a placeholder spec.
    config_specs: list[ir.Module] = [
        spec_builder.get_placeholder_spec(input_module.context)
    ]

    # Get GPU target information from the executable variant operation.
    variant_op_list = iree_codegen.get_executable_variant_ops(input_module)
    assert len(variant_op_list) == 1, "Expect one executable variant op"
    variant_op = variant_op_list[0]
    executable_variant_op = variant_op.opview
    target = executable_variant_op.target
    target_info = iree_gpu.TargetInfo.get_gpu_target_info(target)

    if target_info.arch not in ["gfx942", "gfx1100"]:
        print(f"Warning: Untested architecture '{target_info.arch}'.")

    constraint_generator = dispatch_tuner.get_constraint_generator()

    for i, config in enumerate(
        constraint_generator.generate_solutions(
            tuner_context,
            target_info,
            codegen_pipeline,
            num_subgroups=num_subgroups,
            allowed_waves_per_eu=allowed_waves_per_eu,
            pipeline_options_search_space=pipeline_options_search_space,
        )
    ):
        if i >= limit:
            break
        tune_logger.debug(f"Solution #{i+1}: {config}")
        td_spec_module = dispatch_tuner.get_td_spec(config)
        assert td_spec_module, "Failed to generate transform dialect spec"
        config_specs.append(td_spec_module)

    tune_logger.debug(f"Generated {len(config_specs)} tuning specs")
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
    root_ops: list[ir.Operation] = iree_codegen.get_tuner_root_ops(module)
    for root_op in root_ops:
        assert (
            spec_builder.ROOT_OP_ATTR_NAME in root_op.opview.attributes
        ), f"expected root op to have '{spec_builder.ROOT_OP_ATTR_NAME}' attr"
        del root_op.opview.attributes[spec_builder.ROOT_OP_ATTR_NAME]


# See the above comment for `strip_root_op_attr`.
def strip_compilation_info(input_path: Path) -> str:
    # Strip compilation info from the source and save the stripped IR
    iree_opt = ireec.binaries.find_tool("iree-opt")
    strip_command = [
        iree_opt,
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


def main() -> None:
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
        "--prefetch-shared-memory-options",
        type=lambda t: [s.strip().lower() == "true" for s in t.split(",")],
        default=[True],
        help="Comma-separated list of allowed values for the prefetch_shared_memory pipeline option. Possible values: [True, False]",
    )
    parser.add_argument(
        "--no-reduce-shared-memory-bank-conflicts-options",
        type=lambda t: [s.strip().lower() == "true" for s in t.split(",")],
        default=[None],
        help="Comma-separated list of allowed values for the no_reduce_shared_memory_bank_conflicts pipeline option. Possible values: [True, False]",
    )
    parser.add_argument(
        "--waves-per-eu-options",
        type=lambda t: [int(s) for s in t.split(",")],
        default=[2],
        help="Comma-separated list of allowed values for the waves_per_eu config option. Possible values: Any positive integer value",
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

    with common.TunerContext() as tuner_ctx:
        mlir_text = strip_compilation_info(args.input)
        mlir_module = dispatch_parser.parse_mlir(mlir_text, tuner_ctx)
        pipeline_options_search_space = dispatch_constraints.PipelineOptionsSearchSpace(
            prefetch_shared_memory=args.prefetch_shared_memory_options,
            no_reduce_shared_memory_bank_conflicts=args.no_reduce_shared_memory_bank_conflicts_options,
        )
        specs: list[ir.Module] = generate_configs_and_td_specs(
            mlir_module,
            tuner_ctx,
            args.limit,
            args.num_subgroups,
            args.waves_per_eu_options,
            pipeline_options_search_space,
            iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
        )
        for candidate_num, spec in enumerate(specs):
            spec_dir = Path(args.output)
            spec_path = spec_dir / f"{candidate_num}_spec.mlir"
            spec_dir.mkdir(parents=True, exist_ok=True)
            with open(spec_path, "w") as f:
                local_scope_spec_str: str = spec.operation.get_asm(use_local_scope=True)
                f.write(local_scope_spec_str)


if __name__ == "__main__":
    main()
