# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This code implements matcher functions for MLIR modules using python bindings.

from abc import ABCMeta, abstractmethod

from .common import *
from iree.compiler import ir  # type: ignore


class OpMatcher(metaclass=ABCMeta):
    @abstractmethod
    def match(self, op: ir.Operation) -> bool:
        """Check if the op passes the matching criteria."""
        pass


def walk_collect_ops(
    op: ir.Operation,
    ops: list[ir.Operation],
    fn,
) -> ir.WalkResult:
    if fn(op):
        ops.append(op)
    return ir.WalkResult.ADVANCE


def get_ops_from_module(module: ir.Module, fn):
    ops: list[ir.Operation] = []
    for op in module.body.operations:
        op.walk(
            lambda op: walk_collect_ops(op, ops, fn),
            ir.WalkOrder.POST_ORDER,
        )
    return ops


ROOT_OP_ATTR_NAME = "root_op"


def is_root_op(op: ir.Operation) -> bool:
    for attr in op.opview.attributes:
        if attr.name == ROOT_OP_ATTR_NAME:
            return True
    return False


def match_root_op(
    ir_module: ir.Module,
    matcher: OpMatcher,
) -> Optional[ir.Operation]:
    root_ops: list[ir.Operation] = get_ops_from_module(ir_module, is_root_op)
    if len(root_ops) != 1:
        return None
    if not matcher.match(root_ops[0].operation):
        return None
    return root_ops[0]


class NamedOpMatcher(OpMatcher):
    def __init__(self, op_names: list[str]):
        self.op_names = op_names

    def match(self, op: ir.Operation) -> bool:
        return op.name in self.op_names


# TODO(Max191): Add logic to match the body of the generic op.
class GenericOpMatcher(NamedOpMatcher):
    def __init__(self):
        super().__init__(["linalg.generic"])

    @abstractmethod
    def match_operands(self, operands: ir.OpOperandList) -> bool:
        """Match the operands of the linalg op."""
        pass

    @abstractmethod
    def match_indexing_maps(self, maps: list[ir.AffineMap]) -> bool:
        """Match the indexing_maps of the linalg op."""
        pass

    def match(self, op: ir.Operation) -> bool:
        if not super().match(op):
            return False

        if not self.match_operands(op.operands):
            return False

        maps_attr = None
        for attr in op.opview.attributes:
            if attr.name == "indexing_maps" and isinstance(attr.attr, ir.ArrayAttr):
                maps_attr = attr.attr
        if maps_attr is None:
            return False

        maps: list[ir.AffineMap] = []
        for map in maps_attr:
            maps.append(map.value)
        if not self.match_indexing_maps(maps):
            return False

        return True


def get_map_result_dim_positions(map: ir.AffineMap):
    exprs = []
    if not map.is_projected_permutation:
        return None
    for expr in map.results:
        dim_str = str(expr)
        if len(dim_str) < 1:
            return None
        if dim_str[0] != "d":
            return None
        if not dim_str[1:].isdigit():
            return None
        dim_position = int(dim_str[1:])
        exprs.append(dim_position)
    return exprs


class ContractionOpInterfaceMatcher(GenericOpMatcher):
    def __init__(self) -> None:
        super().__init__()
        self.contraction_dimensions: Optional[ContractionDimensions] = None
        self.lhs_dims: Optional[list[int]] = None
        self.rhs_dims: Optional[list[int]] = None
        self.res_dims: Optional[list[int]] = None

    def match_operands(self, operands: ir.OpOperandList) -> bool:
        if len(operands) != 3:
            return False
        for operand in operands:
            if not isinstance(operand.type, ir.ShapedType):
                return False
        return True

    def match_indexing_maps(self, maps: list[ir.AffineMap]) -> bool:
        if len(maps) != 3:
            return False
        lhs_dims = get_map_result_dim_positions(maps[0])
        rhs_dims = get_map_result_dim_positions(maps[1])
        res_dims = get_map_result_dim_positions(maps[2])
        if lhs_dims is None or rhs_dims is None or res_dims is None:
            return False

        batch_dims = []
        m_dims = []
        n_dims = []
        k_dims = []

        for d in range(maps[0].n_dims):
            if d in lhs_dims and d in rhs_dims and d in res_dims:
                batch_dims.append(d)
                continue
            if d in lhs_dims and d in res_dims:
                m_dims.append(d)
                continue
            if d in rhs_dims and d in res_dims:
                n_dims.append(d)
                continue
            if d in lhs_dims and d in rhs_dims:
                k_dims.append(d)
                continue
            return False

        self.contraction_dimensions = ContractionDimensions(
            m=m_dims,
            n=n_dims,
            k=k_dims,
            batch=batch_dims,
        )
        self.lhs_dims = lhs_dims
        self.rhs_dims = rhs_dims
        self.res_dims = res_dims
        return True
