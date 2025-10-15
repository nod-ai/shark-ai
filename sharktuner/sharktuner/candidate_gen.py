# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import logging
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional, Iterator
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


def set_dispatch_tuner(input_module: ir.Module) -> DispatchTuner:
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
        assert False, "No root ops found"
    elif len(root_op_list) > 1:
        tune_logger.error("Multiple root ops found. Only one is currently supported.")
        assert False, "Multiple root ops found"

    root_op = root_op_list[0]

    dispatch_tuner: Optional[DispatchTuner] = None
    for tuner_class in dispatch_tuners:
        tuner = tuner_class(root_op)
        if tuner.has_valid_root_op():
            dispatch_tuner = tuner
            break

    assert dispatch_tuner, "No suitable dispatch tuner found"
    return dispatch_tuner


def generate_solutions(
    dispatch_tuner: DispatchTuner,
    input_module: ir.Module,
    tuner_context: common.TunerContext,
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints.
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
) -> Iterator[list[common.TuningConfiguration]]:
    # Get GPU target information from the executable variant operation.
    variant_op_list = iree_codegen.get_executable_variant_ops(input_module)
    assert len(variant_op_list) == 1, "Expect one executable variant op"
    variant_op = variant_op_list[0]
    executable_variant_op = variant_op.opview
    target = executable_variant_op.target
    target_info = iree_gpu.TargetInfo.get_gpu_target_info(target)

    if target_info.arch not in ["gfx942", "gfx950", "gfx1100", "gfx1201"]:
        print(f"Warning: Untested architecture '{target_info.arch}'.")

    constraint_generator = dispatch_tuner.get_constraint_generator()

    return constraint_generator.generate_solutions(
        tuner_context,
        target_info,
        codegen_pipeline,
        num_subgroups=num_subgroups,
        allowed_waves_per_eu=allowed_waves_per_eu,
        pipeline_options_search_space=pipeline_options_search_space,
    )


def generate_configs_and_td_specs(
    dispatch_tuner: DispatchTuner,
    input_module: ir.Module,  # In-memory module to be tuned.
    solutions: list[list[common.TuningConfiguration]],
) -> list[ir.Module]:
    # Index 0 is reserved for default config, so it gets a placeholder spec.
    config_specs: list[ir.Module] = [
        spec_builder.get_placeholder_spec(input_module.context)
    ]

    for i, config in enumerate(solutions):
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
    timeout_seconds: Optional[float] = None


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
        # Convert the command list to a command string for logging.
        command_str = " ".join(command)
        logging.debug(f"Run: {command_str}")

        # Add timeout to subprocess.run call.
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
    # Strip compilation info from the source and save the stripped IR.
    iree_opt = ireec.binaries.find_tool("iree-opt")  # type: ignore
    assert iree_opt, "iree-opt tool not found"
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
