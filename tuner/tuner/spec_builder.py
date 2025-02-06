# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

import argparse
from pathlib import Path
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, transform  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *
from .op_matchers import ROOT_OP_ATTR_NAME


def get_matcher_calls(op: ir.Operation, matchers: list[ir.Attribute], actions: list[ir.Attribute]):
    if not isinstance(op.opview, transform.ForeachMatchOp):
        return ir.WalkResult.ADVANCE
    matchers.extend(op.opview.attributes["matchers"])
    actions.extend(op.opview.attributes["actions"])
    return ir.WalkResult.INTERRUPT

def get_extra_spec_info(spec_file: Path, used_matcher_names: list[str], used_action_names: list[str]):
    with open(spec_file) as f:
        spec = f.read()
    with ir.Context():
        spec_module = ir.Module.parse(spec)
        matchers: transform.NamedSequenceOp = []
        matcher_calls: list[ir.FlatSymbolRefAttr] = []
        matcher_call_actions: list[ir.FlatSymbolRefAttr] = []
        for op in spec_module.body.operations:
            if not isinstance(op, transform.NamedSequenceOp):
                continue
            if op.sym_name.value == "__kernel_config":
                op.walk(
                    lambda op: get_matcher_calls(op, matcher_calls, matcher_call_actions),
                    ir.WalkOrder.POST_ORDER,
                )
                continue
            matchers.append(op)
        filtered_matcher_calls = []
        filtered_matcher_call_actions = []
        for call, action in zip(matcher_calls, matcher_call_actions):
            if call.value in used_matcher_names:
                continue
            filtered_matcher_calls.append(call)
            filtered_matcher_call_actions.append(action)
        filtered_matchers = [matcher for matcher in matchers if matcher.sym_name.value not in used_matcher_names + used_action_names]
    matcher_calls_str = "\n, ".join([f"{call} -> {action}" for call, action in zip(filtered_matcher_calls, filtered_matcher_call_actions)])
    matchers_str = "\n".join([str(op) for op in filtered_matchers])
    return matchers_str, matcher_calls_str

def get_placeholder_spec(context: ir.Context, args: argparse.Namespace) -> ir.Module:
    extra_matchers, extra_matcher_fn_calls = get_extra_spec_info(
        args.extra_spec_file,
        used_matcher_names=[],
        used_action_names=[],
    )
    if len(extra_matcher_fn_calls) == 0:
        spec_text = f"""
            module attributes {{ transform.with_named_sequence }} {{
                transform.named_sequence
                @__kernel_config(%variant_op: !transform.any_op {{transform.readonly}}) -> !transform.any_op
                    attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
                    transform.yield %variant_op : !transform.any_op
                }}
            }}
            """
    else:
        spec_text = f"""
            module attributes {{ transform.with_named_sequence }} {{
                {extra_matchers}
                transform.named_sequence
                @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) -> !transform.any_op
                    attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
                    %res = transform.foreach_match in %variant_op
                        {extra_matcher_fn_calls}
                    : (!transform.any_op) -> (!transform.any_op)
                    transform.yield %res : !transform.any_op
                }}
            }}
            """
    return ir.Module.parse(spec_text, context)


# TODO(Max191): Use python bindings to build the transform dialect spec module
# instead of using string formatting.
def build_td_spec(
    context: ir.Context,
    op: ir.Operation,
    compilation_info: iree_codegen.CompilationInfoAttr,
    func_name: str,
    args: argparse.Namespace,
) -> ir.Module:
    extra_matchers, extra_matcher_fn_calls = get_extra_spec_info(
        args.extra_spec_file,
        used_matcher_names=[func_name],
        used_action_names=["apply_op_config"],
    )
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
        ssa_name = operand.get_name()
        operand_type = operand.type
        bbargs.append(f"{ssa_name}: {operand_type}")
        captured_values.add(operand)
    bbargs_str = ", ".join(bbargs)
    spec_text = f"""
        module attributes {{ transform.with_named_sequence }} {{
            // Annotation Transform
            transform.named_sequence @apply_op_config(%op: !transform.any_op {{transform.readonly}},
                                                        %config: !transform.any_param {{transform.readonly}}) {{
                transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
                transform.yield
            }}

            // Custom Op Matcher
            transform.named_sequence @{func_name}(%cont: !transform.any_op {{transform.readonly}})
                -> (!transform.any_op, !transform.any_param) {{
                %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {{
                ^bb0({bbargs_str}):
                {root_operation}
                }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
                %config = transform.param.constant {compilation_info} -> !transform.any_param
                transform.yield %cont, %config : !transform.any_op, !transform.any_param
            }}

            {extra_matchers}

            // Entry Point
            transform.named_sequence
            @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) -> !transform.any_op
                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
                %res = transform.foreach_match in %variant_op
                    @{func_name} -> @apply_op_config
                    , {extra_matcher_fn_calls}
                : (!transform.any_op) -> !transform.any_op
                transform.yield %res : !transform.any_op
            }}
        }}
        """
    return ir.Module.parse(spec_text, context)
