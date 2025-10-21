# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import logging
from abc import ABC, abstractmethod

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, preprocessing_transform, transform  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *

ROOT_OP_ATTR_NAME = "root_op"


def get_matcher_named_sequence_name(root_op: ir.Operation) -> str:
    """
    Returns the symbol name for a transform dialect named sequence matchger.
    """
    return f"match_{get_parent_function_name(root_op)}"


def get_placeholder_spec(context: ir.Context) -> ir.Module:
    spec_text = f"""
        module attributes {{ transform.with_named_sequence }} {{
            transform.named_sequence
            @__kernel_config(%variant_op: !transform.any_op {{transform.readonly}}) -> !transform.any_op
                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
                transform.yield %variant_op : !transform.any_op
            }}
        }}
        """
    return ir.Module.parse(spec_text, context)


# TODO(Max191): Use python bindings to build the transform dialect spec module
# instead of using string formatting.
def build_td_spec(
    context: ir.Context,
    op: ir.Operation,
    config_list: list[common.TuningConfiguration],
    func_name: str,
) -> ir.Module:
    bbargs = []
    # The `root_op` attribute will prevent matching of ops without the attr in
    # the resulting TD spec matcher if it is not removed, so we remove it here.
    # After removing, we must add it back, since the op is connected to the
    # input module, which gets used for all candidates.
    # TODO(Max191): Find a cleaner way to do this without removing and adding
    # back the attribute.
    has_root_attr = ROOT_OP_ATTR_NAME in op.opview.attributes
    if has_root_attr:
        assert isinstance(
            op.opview.attributes[ROOT_OP_ATTR_NAME], ir.UnitAttr
        ), f"expected '{ROOT_OP_ATTR_NAME}' attr to be a unit attr"
    if has_root_attr:
        del op.opview.attributes[ROOT_OP_ATTR_NAME]
    # Get the root op string for formatting the final spec.
    root_operation = str(op)
    if has_root_attr:
        op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)

    # Get the names ssa names of operands to make sure they match in the
    # template after string formatting.
    captured_values: set[ir.Value] = set()
    for operand in op.operands:
        if operand in captured_values:
            # TODO(Max191): Remove this warning when the transform for the
            # `cast_compatible_dag_from_root` op fixes a bug in the matching
            # logic that causes failure to match when the same operand is
            # repeated. For now, still avoid adding duplicate SSA values to
            # prevent parsing failure.
            logging.warning(
                f"Root op has repeated operand. This can cause failure to match in the resulting TD spec at compile time."
            )
            continue
        operand_name = operand.get_name()
        operand_type = operand.type
        bbargs.append(f"{operand_name }: {operand_type}")
        captured_values.add(operand)
    bbargs_str = ", ".join(bbargs)
    matcher_block = f"""%ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {{
          ^bb0({bbargs_str}):
          {root_operation}
        }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)"""

    config_lines = []
    yield_vars = []
    for i, config in enumerate(config_list):
        config_var = f"%{config.name}_{i}"
        config_lines.append(
            f"{config_var} = transform.param.constant {config.configuration} -> !transform.any_param"
        )
        yield_vars.append(config_var)
    config_block = "\n                ".join(config_lines)
    yield_list = ", ".join(["%cont"] + yield_vars)
    yield_types = ", ".join(
        ["!transform.any_op"] + ["!transform.any_param"] * len(yield_vars)
    )

    annotation_args = ", ".join(
        f"%cfg_{i}: !transform.any_param {{transform.readonly}}"
        for i in range(len(config_list))
    )
    annotation_lines = "\n".join(
        f'                transform.annotate %op "{config.name}" = %cfg_{i} : !transform.any_op, !transform.any_param'
        for i, config in enumerate(config_list)
    )

    spec_text = f"""\
        module attributes {{ transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint }} {{
        // Annotation Transform
        transform.named_sequence @apply_op_config(%op: !transform.any_op {{transform.readonly}}, {annotation_args}) {{
        {annotation_lines}
            transform.yield
        }}

        // Custom Op Matcher
        transform.named_sequence @{func_name}(%cont: !transform.any_op {{transform.readonly}})
            -> ({yield_types}) {{
            {matcher_block}
            {config_block}
            transform.yield {yield_list} : {yield_types}
        }}

        // Entry Point
        transform.named_sequence
        @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) -> !transform.any_op
            attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
            %res = transform.foreach_match in %variant_op
                @{func_name} -> @apply_op_config
            : (!transform.any_op) -> !transform.any_op
            transform.yield %res : !transform.any_op
        }}
    }}"""
    return ir.Module.parse(spec_text, context)


def get_readonly_arg_attr() -> dict[str, ir.Attribute]:
    return {"transform.readonly": ir.UnitAttr.get()}


def get_consumed_arg_attr() -> dict[str, ir.Attribute]:
    return {"transform.consumed": ir.UnitAttr.get()}


class SpecBuilder(ABC):
    def __init__(self, op_info: OpInfo):
        self.op_info = op_info

    def create_config_params(
        self, config_list: list[common.TuningConfiguration]
    ) -> list[ir.Value]:
        """
        Creates a constant parameter for each configuration.
        Parameters can contain #iree_codegen.compilation_info or other configuration attributes.
        """
        return [
            transform.ParamConstantOp(
                transform.AnyParamType.get(),
                config.configuration,
            ).result
            for config in config_list
        ]

    @abstractmethod
    def build_matcher(
        self,
        entry_block: ir.Block,
        cont_handle: ir.Value,
        config_list: list[common.TuningConfiguration],
    ) -> tuple[ir.OpResult, list[ir.OpResult]]:
        pass

    def create_matcher_sequence(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> transform.NamedSequenceOp:
        """
        Creates a transform.named_sequence that matches the operation and returns
        the matched operation handle along with configuration parameters.
        """
        input_types = [transform.AnyOpType.get()]
        output_types = [transform.AnyOpType.get()] + [
            transform.AnyParamType.get()
        ] * len(config_list)

        named_seq = transform.NamedSequenceOp(
            get_matcher_named_sequence_name(self.op_info.root_op),
            input_types,
            output_types,
            arg_attrs=[get_readonly_arg_attr()],
        )

        with ir.InsertionPoint(named_seq.body):
            matched_op, config_params = self.build_matcher(
                named_seq.body, named_seq.bodyTarget, config_list
            )

            transform.YieldOp([matched_op] + config_params)

        return named_seq

    def create_annotation_sequence(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> transform.NamedSequenceOp:
        """
        Creates a transform.named_sequence that annotates an operation with
        configuration parameters.
        """
        input_types = [transform.AnyOpType.get()] + [
            transform.AnyParamType.get()
        ] * len(config_list)
        output_types: list[ir.Type] = []

        named_seq = transform.NamedSequenceOp(
            "apply_op_config",
            input_types,
            output_types,
            arg_attrs=[get_readonly_arg_attr()] * len(input_types),
        )

        with ir.InsertionPoint(named_seq.body):
            op_handle = named_seq.bodyTarget
            config_params = list(named_seq.body.arguments)[1:]

            for i, config in enumerate(config_list):
                transform.AnnotateOp(
                    op_handle,
                    config.name,
                    param=config_params[i],
                )

            transform.YieldOp([])

        return named_seq

    def create_entrypoint_sequence(
        self,
    ) -> transform.NamedSequenceOp:
        """
        Creates the @__kernel_config entrypoint sequence.
        """
        input_types = [transform.AnyOpType.get()]
        output_types = [transform.AnyOpType.get()]

        named_seq = transform.NamedSequenceOp(
            "__kernel_config",
            input_types,
            output_types,
            arg_attrs=[get_consumed_arg_attr()],
        )
        named_seq.operation.attributes[
            "iree_codegen.tuning_spec_entrypoint"
        ] = ir.UnitAttr.get()

        with ir.InsertionPoint(named_seq.body):
            variant_op = named_seq.bodyTarget

            result = transform.ForeachMatchOp(
                transform.AnyOpType.get(),
                [],
                variant_op,
                [],
                [get_matcher_named_sequence_name(self.op_info.root_op)],
                ["apply_op_config"],
            ).updated

            transform.YieldOp([result])

        return named_seq

    def build_td_spec(
        self,
        tuner_ctx: common.TunerContext,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        """
        Builds a td spec module using Python bindings.
        """
        context = tuner_ctx.mlir_ctx
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            module.operation.attributes[
                "transform.with_named_sequence"
            ] = ir.UnitAttr.get()
            module.operation.attributes[
                "iree_codegen.tuning_spec_with_default_entrypoint"
            ] = ir.UnitAttr.get()

            with ir.InsertionPoint(module.body):
                self.create_annotation_sequence(config_list)
                self.create_matcher_sequence(config_list)
                self.create_entrypoint_sequence()

            return module


class ContractionSpecBuilder(SpecBuilder):
    def __init__(self, op_info: ContractionOpInfo):
        super().__init__(op_info)
        self.op_info: ContractionOpInfo = op_info

    def build_matcher(
        self,
        entry_block: ir.Block,
        body_target: ir.Value,
        config_list: list[common.TuningConfiguration],
    ) -> tuple[ir.Value, list[ir.Value]]:
        """
        Gets a contraction matcher using transform.iree.match.contraction.
        """
        lhs_elem_type = self.op_info.lhs_type.element_type
        rhs_elem_type = self.op_info.rhs_type.element_type
        res_elem_type = self.op_info.res_type.element_type

        m_dims = self.op_info.matmul_size.M
        n_dims = self.op_info.matmul_size.N
        k_dims = self.op_info.matmul_size.K
        batch_dims = self.op_info.matmul_size.B

        with ir.InsertionPoint(entry_block):
            batch, m, n, k = preprocessing_transform.MatchContractionOp(
                operand_handle=body_target,
                lhs_type=lhs_elem_type,
                rhs_type=rhs_elem_type,
                output_type=res_elem_type,
                indexing_maps=self.op_info.indexing_maps,
            )

            preprocessing_transform.MatchDimsEqualOp(batch, batch_dims)
            preprocessing_transform.MatchDimsEqualOp(m, m_dims)
            preprocessing_transform.MatchDimsEqualOp(n, n_dims)
            preprocessing_transform.MatchDimsEqualOp(k, k_dims)

            config_params = self.create_config_params(config_list)
            return body_target, config_params


# TODO (Bangtian): This is a temporary solution, a specific path for using Python bindings
# to build td specs for contraction ops.
def build_contraction_td_spec(
    tuner_ctx: common.TunerContext,
    op_info: ContractionOpInfo,
    config_list: list[common.TuningConfiguration],
) -> ir.Module:
    """
    Python bindings-based td spec builder for contraction operations.
    """
    builder = ContractionSpecBuilder(op_info)
    return builder.build_td_spec(tuner_ctx, config_list)
