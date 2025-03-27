# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest merge_td_specs_test.py
"""

import pytest

from typing import Generator

from iree.compiler import ir  # type: ignore

from . import merge_td_specs
from . import common


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    from logging import Logger
    from unittest.mock import MagicMock

    mock_logger = MagicMock(spec=Logger)
    with common.TunerContext(logger=mock_logger) as ctx:
        yield ctx


def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    first_module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence } {
        }
    """

    second_module_str = """
        module @inner_module_b
            attributes { transform.with_named_sequence } {
        }
    """

    first_ir_module = ir.Module.parse(first_module_str, context)
    second_ir_module = ir.Module.parse(second_module_str, context)

    module = merge_td_specs.combine_tuning_specs(
        tuner_ctx, [first_ir_module, second_ir_module]
    )
    assert module
    assert "transform.with_named_sequence" in module.operation.attributes

    inner_ops = list(module.body.operations)
    assert all(
        op.name == "builtin.module" for op in inner_ops
    ), "Not all ops are builtin.module"
    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
    assert (
        inner_ops[0].sym_name.value == "inner_module_a"
    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
    assert (
        inner_ops[1].sym_name.value == "inner_module_b"
    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"


def test_merge_tuning_specs(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg : !transform.any_op
            }

            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
                    : (!transform.any_op) -> (!transform.any_op)
                transform.yield %res : !transform.any_op
            }
        }
    """

    first_ir_module = ir.Module.parse(module_str, context)
    second_ir_module = ir.Module.parse(module_str, context)
    second_ir_module.operation.attributes["sym_name"] = ir.StringAttr.get(
        "inner_module_b"
    )
    merged_module = merge_td_specs.merge_tuning_specs(
        tuner_ctx, [first_ir_module, second_ir_module]
    )
    assert merged_module

    assert "transform.with_named_sequence" in merged_module.operation.attributes
    assert (
        "iree_codegen.tuning_spec_with_default_entrypoint"
        in merged_module.operation.attributes
    )

    inner_ops = list(merged_module.body.operations)
    # Check that inner modules have been merged into the top-level module and no inner modules remain.
    assert all(
        op.name != "builtin.module" for op in inner_ops
    ), "Unexpected inner builtin.module ops found"

    named_sequences = []
    kernel_config_op = None
    for op in merged_module.body.operations:
        if op.name == "transform.named_sequence":
            sym_name_attr = op.sym_name
            assert sym_name_attr is not None
            named_sequences.append(sym_name_attr.value)
            if sym_name_attr.value == "__kernel_config":
                kernel_config_op = op

    assert kernel_config_op is not None, "Missing @__kernel_config"


def test_merge_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence } {
        }
    """

    module = ir.Module.parse(module_str, context)
    module.operation.attributes[
        "iree_codegen.tuning_spec_with_default_entrypoint"
    ] = ir.UnitAttr.get()
    with pytest.raises(RuntimeError) as exc_info:
        merge_td_specs.merge_tuning_specs(tuner_ctx, [module])
        # iree-opt should fail due to missing named sequence @__kernel_config entrypoint required
        # by the `iree_codegen.tuning_spec_with_default_entrypoint` attribute.
        assert "iree-opt failed" in str(exc_info.value)
