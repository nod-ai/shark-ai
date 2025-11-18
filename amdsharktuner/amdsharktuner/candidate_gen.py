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
from iree.compiler.dialects import iree_codegen, iree_gpu, linalg  # type: ignore

from . import (
    common,
    dispatch_constraints,
    dispatch_parser,
    spec_builder,
    constraint_generator,
)

tune_logger = logging.getLogger("tune")


class DispatchTuner(dispatch_parser.DispatchParser):
    @classmethod
    @abstractmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        """Check if this tuner can handle the given root operation."""
        pass

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

    @classmethod
    @abstractmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        """Returns dispatch kind"""
        pass

    @abstractmethod
    def get_knob_assignment(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        """
        Return a KnobAssignment that records the feature values of a single candidate,
        retrieved from the `knob_assignment` attribute of its TuningConfiguration.
        """
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
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_contraction_op(root_op):
            return False

        # Check if contraction has valid dimensions.
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        if not contraction_dims:
            logging.warning("No contraction dimensions found for operation")
            return False

        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
            logging.warning(
                f"Contraction operation with dimensions M={list(contraction_dims.m)}, "
                f"N={list(contraction_dims.n)}, K={list(contraction_dims.k)} "
                f"is not supported by the tuner yet"
            )
            return False

        return True

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return constraint_generator.ContractionOpInterfaceConstraintGenerator(
            self.get_op_info()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ContractionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.contraction

    def get_knob_assignment(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return config_list[0].knob_assignment


class ConvolutionOpInterfaceTuner(
    DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_convolution_op(root_op):
            return False
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        if not convolution_dims:
            return False
        # Only allow 'nhwc_hwcf' convs.
        return (
            list(convolution_dims.batch) == [0]
            and list(convolution_dims.output_image) == [1, 2]
            and list(convolution_dims.output_channel) == [3]
            and list(convolution_dims.filter_loop) == [4, 5]
            and list(convolution_dims.input_channel) == [6]
            and list(convolution_dims.depth) == []
        )

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return constraint_generator.ConvolutionOpInterfaceConstraintGenerator(
            self.get_op_info()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ConvolutionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.conv

    def get_knob_assignment(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return None


class AttentionOpInterfaceTuner(
    DispatchTuner, dispatch_parser.AttentionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        return iree_codegen.isa_attention_op(root_op)

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return constraint_generator.AttentionOpInterfaceConstraintGenerator(
            self.get_op_info()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.AttentionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.attention

    def get_knob_assignment(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return None


def set_dispatch_tuner(
    input_module: ir.Module, tuner_ctx: common.TunerContext
) -> Optional[DispatchTuner]:
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
        return None
    elif len(root_op_list) > 1:
        tune_logger.error("Multiple root ops found. Only one is currently supported.")
        return None

    root_op = root_op_list[0]

    dispatch_tuner: Optional[DispatchTuner] = None
    for tuner_class in dispatch_tuners:
        if tuner_class.supports_root_op(root_op):
            tuner = tuner_class(root_op, tuner_ctx)
            dispatch_tuner = tuner
            break

    if not dispatch_tuner:
        tune_logger.error(
            "No suitable dispatch tuner found for the root operation. "
            "The operation may not be supported by the tuner yet."
        )

    return dispatch_tuner


def generate_solutions(
    dispatch_tuner: DispatchTuner,
    target_info: iree_gpu.TargetInfo,
    tuner_context: common.TunerContext,
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints.
    allowed_waves_per_eu: list[int] = [2],
    pipeline_options_search_space: dispatch_constraints.PipelineOptionsSearchSpace = dispatch_constraints.PipelineOptionsSearchSpace(),
    codegen_pipeline: iree_codegen.DispatchLoweringPassPipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
) -> Iterator[list[common.TuningConfiguration]]:
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
