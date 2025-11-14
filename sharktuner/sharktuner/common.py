# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, asdict
from enum import Enum
from types import TracebackType
from typing import Optional, Any
from abc import ABC
import os
import time
import z3  # type: ignore
import subprocess
import tempfile

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu, transform  # type: ignore
import iree.compiler as ireec  # type: ignore
from iree.compiler._mlir_libs._mlir import ir  # type: ignore


class CommonTypes:
    def __init__(self, ctx: ir.Context):
        assert ctx
        self.i1 = ir.IntegerType.get_signless(1, ctx)
        self.i8 = ir.IntegerType.get_signless(8, ctx)
        self.i16 = ir.IntegerType.get_signless(16, ctx)
        self.i32 = ir.IntegerType.get_signless(32, ctx)
        self.i64 = ir.IntegerType.get_signless(64, ctx)

        self.f8E4M3FNUZ = ir.Float8E4M3FNUZType.get(ctx)
        self.f8E5M2FNUZ = ir.Float8E5M2FNUZType.get(ctx)
        self.f16 = ir.F16Type.get(ctx)
        self.f32 = ir.F32Type.get(ctx)

        self.bf16 = ir.BF16Type.get(ctx)

    def getI64(self, value: int) -> ir.IntegerAttr:
        return ir.IntegerAttr.get(self.i64, value)

    def getI64ArrayAttr(self, values: list[int]) -> ir.ArrayAttr:
        return ir.ArrayAttr.get([self.getI64(x) for x in values])


class TunerContext:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.mlir_ctx: ir.Context = ir.Context()
        self.logger: logging.Logger = logger or logging.getLogger("tune")
        self.type: CommonTypes = CommonTypes(self.mlir_ctx)

    def __enter__(self) -> "TunerContext":
        self.mlir_ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self.mlir_ctx.__exit__(exc_type, exc_value, traceback)


@dataclass
class TimeBudget:
    """Wall-clock deadline helper based on time.monotonic()."""

    deadline: Optional[float] = None  # Absolute monotonic time (seconds).

    @classmethod
    def for_minutes(cls, minutes: Optional[float], now: Optional[float] = None):
        """Create a budget that lasts 'minutes' from a given 'now' (monotonic seconds)."""
        if minutes is None or minutes <= 0:
            return None
        if now is None:
            now = time.monotonic()
        return cls(now + (minutes * 60.0))

    def expired(self, current_time: Optional[float] = None) -> bool:
        if current_time is None:
            current_time = time.monotonic()
        return self.deadline is not None and current_time >= self.deadline

    def remaining(self, current_time: Optional[float] = None) -> Optional[float]:
        if current_time is None:
            current_time = time.monotonic()
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - current_time)


@dataclass
class KnobAssignment(ABC):
    """A KnobAssignment is a record of tuning parameters values from constraint_generator"""

    def get_knobs(self) -> dict:
        """Return a dict of all knob parameters and their assigned values."""
        return asdict(self)


@dataclass
class TuningConfiguration:
    """
    A TuningConfiguration contains an attribute that will be set on an op as a
    result of running a tuning spec, along with its name. For example, a common
    tuning configuration would have "compilation_info" as its name, and an
    `iree_codegen.CompilationInfoAttr` as the configuration.

    Example:
        TuningConfiguration(name="compilation_info", configuration=CompilationInfoAttr(...))
    """

    name: str
    configuration: ir.Attribute
    knob_assignment: Optional[KnobAssignment] = None


class DispatchKind(Enum):
    conv = 0
    contraction = 1
    attention = 2


@dataclass
class ShapedType:
    shape: list[int]
    element_type: ir.IntegerType | ir.FloatType

    def rank(self) -> int:
        return len(self.shape)

    @property
    def bitwidth(self) -> int:
        return self.element_type.width

    def __str__(self) -> str:
        dim_to_str = lambda dim: str(dim) if dim != -1 else "?"
        return "x".join(map(dim_to_str, self.shape)) + "x" + str(self.element_type)


@dataclass
class ContractionSizes:
    """
    Represents the size of the iteration space along each contraction dimension.
    For example, the following is a simple batch mmt:
      linalg.generic ... indexing_maps = [
          affine_map<(b, m, n, k) -> (b, m, k)>,
          affine_map<(b, m, n, k) -> (b, n, k)>,
          affine_map<(b, m, n, k) -> (b, m, n)>,
        ] ...
        ins(%lhs: tensor<4x8x32xf16>, %rhs: tensor<4x16x32xf16>)
        outs(%acc: tensor<4x8x16xf16>)
    The ContractionSizes would be:
      M = [8]
      N = [16]
      K = [32]
      B = [4]
    """

    M: list[int]
    N: list[int]
    K: list[int]
    B: list[int] = field(default_factory=list)


@dataclass
class ContractionDimensions:
    """
    Stores which dimensions of the iteration space belong to M, N, K, or Batch.
    For example, the following is a simple batch mmt:
    linalg.generic ... indexing_maps = [
        affine_map<(b, m, n, k) -> (b, m, k)>,
        affine_map<(b, m, n, k) -> (b, n, k)>,
        affine_map<(b, m, n, k) -> (b, m, n)>,
        ]
    The ContractionDimensions would be:
    M = [1]
    N = [2]
    K = [3]
    B = [0]
    """

    m: list[int]
    n: list[int]
    k: list[int]
    batch: list[int] = field(default_factory=list)


@dataclass
class MatmulShapeType:
    m: int
    n: int
    k: int
    lhs_type: ir.IntegerType | ir.FloatType
    rhs_type: ir.IntegerType | ir.FloatType
    acc_type: ir.IntegerType | ir.FloatType


@dataclass
class LLVMGPUVectorDistributeContractionKnobs(KnobAssignment):
    # Problem Size.
    M: int
    N: int
    K: int

    # Z3 numeric selections.
    tile_m: int
    tile_n: int
    tile_k: int
    wg_x: int
    wg_y: int
    wg_z: int
    subgroup_m_cnt: int
    subgroup_n_cnt: int
    intrinsic_mn: int
    intrinsic_k: int
    subgroup_m: int
    subgroup_n: int
    subgroup_k: int


@dataclass
class ConvolutionKnobs(KnobAssignment):
    pass


@dataclass
class AttentionKnobs(KnobAssignment):
    pass


def get_map_result_dim_positions(map: ir.AffineMap) -> Optional[list[int]]:
    if not map.is_projected_permutation:
        return None

    return [ir.AffineDimExpr(expr).position for expr in map.results]


def get_compatible_mfma_intrinsics(
    lhs_type: ShapedType,
    rhs_type: ShapedType,
    res_type: ShapedType,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
) -> list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic]:
    def is_compatible(
        mma: iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic,
    ) -> bool:
        if isinstance(mma, iree_gpu.VirtualMMAIntrinsic):
            mma_attr = iree_gpu.VirtualMMAAttr.get(mma)
        else:
            mma_attr = iree_gpu.MMAAttr.get(mma)

        a_type, b_type, c_type = mma_attr.abc_element_types
        return (
            lhs_type.element_type == a_type
            and rhs_type.element_type == b_type
            and res_type.element_type == c_type
        )

    return list(filter(is_compatible, mma_intrinsics))


# The key name for GPUPipelineOptionsAttr in the translation info config dictionary.
GPU_PIPELINE_OPTIONS_KEY = "gpu_pipeline_options"
# The key name for llvm_func_attrs attribute in the translation info config dictionary.
LLVM_FUNC_ATTRS_KEY = "llvm_func_attrs"
# The Key name for the 'amdgpu-waves-per-eu' within the llvm_func_attrs attribute.
WAVES_PER_EU_KEY = "amdgpu-waves-per-eu"


def get_lowering_config(
    tuner_ctx: TunerContext,
    **kwargs: Any,
) -> iree_gpu.LoweringConfigAttr:
    lowering_config_dict: dict[str, Any] = {}
    for key, value in kwargs.items():
        # A local variable to hold the transformed value.
        promoted_value = value
        match key:
            case "workgroup" | "reduction" | "subgroup" | "promote_operands" | "padding":
                if isinstance(value, Sequence):
                    promoted_value = ir.ArrayAttr.get(
                        [tuner_ctx.type.getI64(x) for x in value]
                    )
                elif not isinstance(value, ir.ArrayAttr):
                    assert (
                        False
                    ), f"Unsupported type for key '{key}': {type(value).__name__}"
            case "subgroup_basis":
                if isinstance(value, list) and len(value) == 2:
                    counts, mapping = value
                    assert isinstance(counts, list) and isinstance(
                        mapping, list
                    ), f"subgroup_basis must contain two lists [counts, mapping]"
                    counts_attr = tuner_ctx.type.getI64ArrayAttr(counts)
                    mapping_attr = tuner_ctx.type.getI64ArrayAttr(mapping)
                    promoted_value = ir.ArrayAttr.get([counts_attr, mapping_attr])

                else:
                    assert (
                        False
                    ), f"Unsupported type for key '{key}': {type(value).__name__}"
            case "mma_kind":
                if not isinstance(value, (iree_gpu.MMAAttr, iree_gpu.VirtualMMAAttr)):
                    assert (
                        False
                    ), f"Unsupported type for key '{key}': {type(value).__name__}"
            case _:
                assert False, f"Unhandled key in lowering configuration: {key}"

        lowering_config_dict[key] = promoted_value
    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
    return iree_gpu.LoweringConfigAttr.get(lowering_config_attrs)


# Generate a config dictionary used in translation_info attribute.
def get_translation_info_config(
    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int
) -> ir.DictAttr:
    """
    Example IR
    translation_info = #iree_codegen.translation_info<
                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
                     llvm_func_attrs = {"amdgpu-waves-per-eu" = "3"}
                    }
                >
    """
    waves_per_eu_str = str(waves_per_eu)

    # Create the waves_per_eu dictionary attribute.
    waves_per_eu_dict = ir.DictAttr.get(
        {WAVES_PER_EU_KEY: ir.StringAttr.get(waves_per_eu_str)}
    )

    config_dict = ir.DictAttr.get(
        {
            GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
            LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
        }
    )

    return config_dict


def combine_tuning_specs(
    tuner_ctx: TunerContext, td_specs: list[ir.Module]
) -> ir.Module:
    """
    Puts multiple input modules `td_specs` into a single top-level container module.
    This function does *not* attempt to merge or link `td_specs` across modules.
    """
    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
        top_module = ir.Module.create()
        top_module.operation.attributes[
            "transform.with_named_sequence"
        ] = ir.UnitAttr.get()

        for td_spec in td_specs:
            top_module.body.append(td_spec.operation.clone())
        return top_module


def link_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
    """
    Links multiple input modules (`td_specs`) into a single tuning specification module.
    First, the input modules are combined into a container module. Then, the external
    `iree-opt` tool is invoked with the `--iree-codegen-link-tuning-specs` pass to
    link or merge the individual tuning specs. When all input specs are marked with the
    default attribute `iree_codegen.tuning_spec_with_default_entrypoint`, they are merged
    into one tuning spec.
    """
    module = combine_tuning_specs(tuner_ctx, td_specs)
    iree_opt = ireec.binaries.find_tool("iree-opt")  # type: ignore
    assert iree_opt, "iree-opt tool not found"

    if len(td_specs) == 1:
        # avoid unnecessary link overhead.
        return td_specs[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "tmp_input.mlir")
        output_path = os.path.join(tmpdir, "tmp_output.mlir")

        with open(input_path, "w") as f:
            f.write(str(module))

        result = subprocess.run(
            [
                iree_opt,
                "--iree-codegen-link-tuning-specs",
                input_path,
                "-o",
                output_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"iree-opt failed: {result.stderr}")

        with open(output_path, "r") as f:
            output_mlir = f.read()
            return ir.Module.parse(output_mlir, tuner_ctx.mlir_ctx)


def get_matcher_names_from_td_spec(td_spec: ir.Module) -> set[str]:
    matcher_names = set()

    for op in td_spec.body.operations:
        if not isinstance(op, transform.NamedSequenceOp):
            continue
        if op.sym_name.value != "__kernel_config":
            continue

        for inner_op in op.regions[0].blocks[0].operations:
            if isinstance(inner_op, transform.ForeachMatchOp):
                for matcher in inner_op.matchers:
                    matcher_names.add(matcher.value)

    return matcher_names


def get_matcher_overlap_info(
    starter_matchers: set[str], current_matchers: set[str]
) -> tuple[set[str], set[str]]:
    """
    Returns:
        - overlapping_matchers: matchers shared by starter and current
        - unique_starter_matchers: matchers only in the starter
    """
    overlapping_matchers = starter_matchers & current_matchers
    unique_starter_matchers = starter_matchers - current_matchers

    return overlapping_matchers, unique_starter_matchers


def determine_td_specs_to_link(
    td_specs: list[ir.Module],
    log_duplicates: bool = False,
) -> list[ir.Module]:
    """
    Determines which tuning specs should be linked based on matcher overlap.

    Args:
        td_specs: A list of 1 or 2 tuning spec modules. If two are provided, the first is
                the candidate spec and the second is the starter spec.
        log_duplicates: If True, logs a warning for overlapping matchers.

    Returns:
        A list of td specs to link (possibly excluding the starter spec).
    """

    assert 1 <= len(td_specs) <= 2, "Expected 1 or 2 td specs (current and starter)"

    if len(td_specs) == 1:
        # No starter td spec provided, nothing to merge.
        return td_specs

    current_td_spec, starter_td_spec = td_specs

    current_matchers = get_matcher_names_from_td_spec(current_td_spec)
    starter_matchers = get_matcher_names_from_td_spec(starter_td_spec)

    overlapping_matchers, unique_starter_matchers = get_matcher_overlap_info(
        starter_matchers, current_matchers
    )

    if log_duplicates and overlapping_matchers:
        logging.warning(
            f"Operations have already been tuned in the starter tuning spec: {sorted(overlapping_matchers)}"
        )

    if unique_starter_matchers:
        return td_specs

    # Starter spec is redundant, so skip merging it.
    return [current_td_spec]


def get_attention_decomposition_config(
    tuner_ctx: TunerContext,
    qk_lowering_config: iree_gpu.LoweringConfigAttr,
    pv_lowering_config: iree_gpu.LoweringConfigAttr,
) -> ir.DictAttr:
    """
    Constructs the decomposition config for an attention op, embedding
    separate lowering configs for QK and PV matmuls.
    """

    ctx = tuner_ctx.mlir_ctx
    qk_attrs_dict = {
        "attention_qk_matmul": ir.UnitAttr.get(ctx),
        "lowering_config": qk_lowering_config,
    }
    qk_attr_dict = ir.DictAttr.get(qk_attrs_dict, context=ctx)

    pv_attrs_dict = {
        "attention_pv_matmul": ir.UnitAttr.get(ctx),
        "lowering_config": pv_lowering_config,
    }
    pv_attr_dict = ir.DictAttr.get(pv_attrs_dict, context=ctx)

    decomposition_config_dict = {
        "qk_attrs": qk_attr_dict,
        "pv_attrs": pv_attr_dict,
    }

    return ir.DictAttr.get(decomposition_config_dict, context=ctx)


def get_target_info(input_module: ir.Module) -> iree_gpu.TargetInfo:
    # Get GPU target information from the executable variant operation.
    variant_op_list = iree_codegen.get_executable_variant_ops(input_module)
    assert len(variant_op_list) == 1, "Expect one executable variant op"
    variant_op = variant_op_list[0]
    executable_variant_op = variant_op.opview
    target = executable_variant_op.target

    return iree_gpu.TargetInfo.get_gpu_target_info(target)
