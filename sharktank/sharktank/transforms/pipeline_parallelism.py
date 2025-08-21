# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import logging
import torch
import torch.fx.passes.split_module

from collections import defaultdict
from enum import Enum
from sharktank.types import AnyTensor
from torch import fx
from torch.distributed.pipelining import Pipe
from torch.export.unflatten import (
    _assign_attr,
    _AttrKind,
    _sink_params,
    InterpreterModule,
    _ModuleFrame,
    _SubmoduleEntry,
)
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

InterStageLiveSet = Sequence[AnyTensor]


class SplitPoint(Enum):
    """
    Enum representing the points at which a split can occur in the execution of a submodule.
    Attributes:
        BEGINNING: Represents adding a split point *before* the execution of a certain submodule in the `forward` function.
        END: Represents adding a split point *after* the execution of a certain submodule in the `forward` function.
    """

    BEGINNING = 1
    END = 2


class Pipe(torch.nn.Module):
    def __init__(
        self,
        split_gm: fx.GraphModule,
        num_stages: int,
    ):
        torch.nn.Module.__init__(self)
        self.split_gm: fx.GraphModule = split_gm
        self.num_stages: int = num_stages


torch.library.define("sharktank::pipeline_split", "() -> ()")


@torch.library.impl("sharktank::pipeline_split", "default")
def _pipeline_split_impl():
    return None


@torch.library.register_fake("sharktank::pipeline_split")
def _pipeline_split_fake():
    return None


# Add an alias for convenience
aten_pipeline_split_alias = torch.ops.sharktank.pipeline_split.default

# Ask Export to preserve the `pipeline_split` op.
# See examples in pytorch/torch/fx/node.py
fx.node._side_effectful_functions.add(aten_pipeline_split_alias)

# User facing API
def pipeline_split():
    """
    pipeline_split is a special operator that is used to mark the boundary between
    stages in a module. It is used to split the module into stages. It is a
    no-op if your annotated module is run eagerly.

    Example:
        >>> # xdoctest: +SKIP
        >>> def forward(self, x):
        >>>     x = torch.mm(x, self.mm_param)
        >>>     x = torch.relu(x)
        >>>     pipeline_split()
        >>>     x = self.lin(x)
        >>>     return x

    The above example will be split into two stages.
    """
    return torch.ops.sharktank.pipeline_split()


def get_submodule_from_path(module: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Retrieves a submodule from a PyTorch nn.Module using a path string.

    Args:
        module (nn.Module): The main PyTorch module.
        path (str): The dot-separated path to the submodule (e.g., "layer1.sub_layer.conv").

    Returns:
        nn.Module: The retrieved submodule.
        None: If the submodule is not found at the specified path.
    """
    current_module = module
    path_components = path.split(".")
    for component in path_components:
        if hasattr(current_module, component):
            current_module = getattr(current_module, component)
        else:
            raise AttributeError(f"Submodule {path} not found.")
    return current_module


def insert_stage_split_markers(
    module: torch.nn.Module,
    *,
    split_spec: dict[tuple[str, str], SplitPoint],
):
    """
    Args:
        split_spec: (submodule_path_name, function_name) -> SplitPoint.
            E.g. ("layers.1", "forward") -> SplitPoint.BEGINNING
    """
    for (submodule_path, function_name), split_point in split_spec.items():
        submodule = get_submodule_from_path(module, submodule_path)
        insert_split_marker(submodule, function_name, split_point)


def insert_split_marker(
    module: torch.nn.Module, function: str, split_point: SplitPoint
):
    fn = getattr(module, function)
    if split_point == SplitPoint.BEGINNING:

        def fn_with_before_marker(self, *args, **kwargs):
            pipeline_split()
            return fn(self, *args, **kwargs)

        setattr(module, function, fn_with_before_marker)
    else:

        def fn_with_after_marker(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            finally:
                pipeline_split()

        setattr(module, function, fn_with_after_marker)


# def _recursive_getattr_with_parent(mod, fqn):
#     # Returns getattr call given a nested FQN, and the last parent
#     atoms = fqn.split(".")
#     for atom in atoms[:-1]:
#         if not hasattr(mod, atom):
#             return None, None
#         mod = getattr(mod, atom)
#     if not hasattr(mod, atoms[-1]):
#         return mod, None
#     attr = getattr(mod, atoms[-1])
#     return mod, attr

# def _move_parameter_to_callee(
#     root: fx.GraphModule,
#     callee_name: str,
#     parameter_name: str,
# ):
#     """
#     Move a parameter from the root module to a submodule.
#     Args:
#         root: The root module.
#         callee_name: The name of the submodule to move the parameter to.
#         parameter_name: The fully qualified name of the parameter to move.
#     """
#     # `atoms` is a list of strings representing the path to the
#     # parameter in the original model
#     atoms = parameter_name.split(".")
#     mod_itr, param_val = _recursive_getattr_with_parent(root, parameter_name)
#     # Check whether the parameter is a buffer or a parameter
#     is_buffer = atoms[-1] in mod_itr._buffers

#     # Check whether the parameter is a tensor
#     assert isinstance(param_val, torch.Tensor), (
#         f"Expected '{parameter_name}' to be {torch.Tensor} but got {type(param_val)}."
#         + (
#             f" It might happen if module '{parameter_name}' was passed to some 'leaf function'"
#             f"(see https://pytorch.org/docs/stable/fx.html#fx.wrap). Please inspect "
#             f"usages of '{parameter_name}' in the traced graph."
#             if isinstance(param_val, torch.nn.Module)
#             else ""
#         )
#     )

#     # Get submodule
#     callee = root.get_submodule(callee_name)
#     assert not hasattr(callee, parameter_name), (
#         f"Module {callee_name} already has a parameter named {parameter_name}"
#     )

#     # Assign the parameter to the submodule
#     if is_buffer:
#         _assign_attr(
#             param_val,
#             callee,
#             parameter_name,
#             attr_kind=_AttrKind.BUFFER,
#             persistent=True,  # TODO: handle non-persistent buffer
#         )
#     else:
#         _assign_attr(
#             param_val,
#             callee,
#             parameter_name,
#             attr_kind=_AttrKind.PARAMETER,
#         )

#     # Next step is to replace placeholder of submodule with a get_attr.
#     # Those placeholders are created by `split_module` inside each
#     # submodule.
#     # Update: this step is now moved to `_sink_params` because
#     # `_sink_params` can do it recursively (i.e. for modules inside
#     # submodule)

#     to_delete.append((mod_itr, atoms[-1]))


def split_annotated_module(
    module: torch.nn.Module,
    *,
    function: str | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    strict: bool = False,
) -> Pipe:
    """Split into stages a module that already has inserted sharktank.pipeline_split
    ops, that denote where to make the cuts."""
    if function is not None and function != "forward":
        raise NotImplementedError("TODO")

    exported_program = torch.export.export(
        module, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=strict
    )

    part_idx = 0

    def split_callback(n: fx.Node):
        nonlocal part_idx
        if (n.op, n.target) == (
            "call_function",
            aten_pipeline_split_alias,
        ):
            part_idx += 1
        return part_idx

    split: fx.GraphModule = torch.fx.passes.split_module.split_module(
        exported_program.module(), module, split_callback
    )
    # a (custom) tracer can produce dead code like orphan get_attr nodes.
    split.graph.eliminate_dead_code()

    for submodule in split.modules():
        if isinstance(submodule, fx.GraphModule):
            for node in submodule.graph.nodes:
                if (node.op, node.target) == (
                    "call_function",
                    aten_pipeline_split_alias,
                ):
                    submodule.graph.erase_node(node)
            submodule.recompile()

    for name, submodule in split.named_children():
        if isinstance(submodule, fx.GraphModule):
            new_submod = _outline_submodules(submodule.graph)
            # Replace old submod
            split.register_module(name, new_submod)

    # TODO: backport this into split_module
    def delete_user_reference(node, user):
        """
        Delete reference of `node` from `user`'s arg list.
        Args:
            - node: a `get_attr` node at root.
            - user: a submodule node that uses `node`.
        """
        assert len(user.kwargs) == 0
        use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
        assert len(use_idxs) == 1
        args_copy = list(user.args)
        args_copy.pop(use_idxs[0])
        user.args = tuple(args_copy)
        logger.debug(
            f"Deleted {node} from user {user}, arg index = {use_idxs[0]}"  # noqa: G004
        )

    # A list of param referrals for deferred deletion.
    # To be accumulated in `move_param_to_callee`.
    to_delete = []

    def _recursive_getattr_with_parent(mod, fqn):
        # Returns getattr call given a nested FQN, and the last parent
        atoms = fqn.split(".")
        for atom in atoms[:-1]:
            if not hasattr(mod, atom):
                return None, None
            mod = getattr(mod, atom)
        if not hasattr(mod, atoms[-1]):
            return mod, None
        attr = getattr(mod, atoms[-1])
        return mod, attr

    def move_param_to_callee(
        root,
        callee_name,
        param_fqn,
    ):
        """
        Move a parameter from the root module to a submodule.
        Args:
            root: The root module.
            callee_name: The name of the submodule to move the parameter to.
            param_fqn: The fully qualified name of the parameter to move.
        """
        # `atoms` is a list of strings representing the path to the
        # parameter in the original model
        atoms = param_fqn.split(".")
        mod_itr, param_val = _recursive_getattr_with_parent(split, param_fqn)
        # Check whether the parameter is a buffer or a parameter
        is_buffer = atoms[-1] in mod_itr._buffers

        # Check whether the parameter is a tensor
        assert isinstance(param_val, torch.Tensor), (
            f"Expected '{param_fqn}' to be {torch.Tensor} but got {type(param_val)}."
            + (
                f" It might happen if module '{param_fqn}' was passed to some 'leaf function'"
                f"(see https://pytorch.org/docs/stable/fx.html#fx.wrap). Please inspect "
                f"usages of '{param_fqn}' in the traced graph."
                if isinstance(param_val, torch.nn.Module)
                else ""
            )
        )

        # Get submodule
        callee = root.get_submodule(callee_name)
        assert not hasattr(
            callee, param_fqn
        ), f"Module {callee_name} already has a parameter named {param_fqn}"

        # Assign the parameter to the submodule
        if is_buffer:
            _assign_attr(
                param_val,
                callee,
                param_fqn,
                attr_kind=_AttrKind.BUFFER,
                persistent=True,  # TODO: handle non-persistent buffer
            )
        else:
            _assign_attr(
                param_val,
                callee,
                param_fqn,
                attr_kind=_AttrKind.PARAMETER,
            )
        logger.debug(f"Moved parameter {param_fqn} to {callee_name}")  # noqa: G004

        # Next step is to replace placeholder of submodule with a get_attr.
        # Those placeholders are created by `split_module` inside each
        # submodule.
        # Update: this step is now moved to `_sink_params` because
        # `_sink_params` can do it recursively (i.e. for modules inside
        # submodule)

        to_delete.append((mod_itr, atoms[-1]))

    # Get the list of all parameters in the root module
    attr_nodes = list(filter(lambda n: n.op == "get_attr", split.graph.nodes))
    for node in attr_nodes:
        # Check whether the parameter is used in only one submodule
        if len(node.users) > 1:
            logger.info(
                f"Parameter {node.target} used in multiple stages: {node.users}."  # noqa: G004
            )
        for user in node.users:
            assert user.op == "call_module"
            # Move parameter into submodule
            move_param_to_callee(
                split,
                user.target,
                node.target,
            )

    # [aliasing] store tensor id -> list of FQNs, built from state dict
    # Also assign non-persistent buffers
    id_to_fqns: dict[int, set[str]] = defaultdict(set)
    for fqn, tensor in module.state_dict(keep_vars=True).items():
        id_to_fqns[id(tensor)].add(fqn)
    for fqn, tensor in module.named_buffers():
        id_to_fqns[id(tensor)].add(fqn)

    # After moving the params to their corresponding hierarchies, we also
    # need to move the `get_attr` nodes from the root of the graph to those
    # hierarchies.
    # [aliasing] use id -> fqn mapping to list out all valid FQNs
    inputs_to_state: dict[str, list[str]] = {}
    for attr in attr_nodes:
        _, tensor = _recursive_getattr_with_parent(module, attr.target)
        fqns = list(id_to_fqns[id(tensor)])
        if fqns:
            inputs_to_state[attr.name] = fqns
        elif attr.target in exported_program.constants:  # lifted constants
            inputs_to_state[attr.name] = [attr.target]

    # [aliasing] for each submodule split, assign attributes on FQNs that may be used.
    # We determine this based on whether or not the FQN attribute parent exists.
    # i.e. if the last submodule exists, assign the attribute.
    added_attributes: dict[str, list[str]] = defaultdict(list)
    for fqn, tensor in module.state_dict(keep_vars=True).items():
        for name, submod in split.named_children():
            if isinstance(submod, fx.GraphModule):
                parent, child = _recursive_getattr_with_parent(submod, fqn)
                if (
                    parent and child is None
                ):  # parent exists, attribute doesn't -> assign
                    added_attributes[name].append(fqn)
                    setattr(parent, fqn.split(".")[-1], tensor)

    # Deferral deletion: Remove the original attributes (to params) from the
    # root GraphModule
    for mod_itr, last_atom in to_delete:
        try:
            delattr(mod_itr, last_atom)
        except AttributeError:
            # This is expected if the parameter is used in multiple stages
            pass

    # This is done by (1) `_sink_params` at each submodule;
    for name, submod in split.named_children():
        if isinstance(submod, fx.GraphModule):
            _sink_params(submod, inputs_to_state, [])
            submod.graph.lint()
            submod.recompile()

    # [aliasing] This step is not super necessary, but helps reduce parameter usage/memory.
    # After _sink_params() routine has run, clean up unused attributes that we previously added.
    # Determine this based on the get_attr nodes - if not used, remove it.
    for name, attributes in added_attributes.items():
        submod = getattr(split, name)
        unused_attributes = set(attributes)
        # track used attributes in the submodule, running DFS on subgraph hierarchy
        stack = [("", submod)]  # (scope, submodule)
        while stack:
            scope, _mod = stack.pop()
            if isinstance(_mod, (fx.GraphModule, InterpreterModule)):
                for node in _mod.graph.nodes:
                    if node.op == "get_attr":
                        # get_attr might get access deeper level attribute
                        fqn = scope + "." + node.target if scope else node.target
                        unused_attributes.discard(fqn)
            for _name, _submod in _mod.named_children():
                stack.append((scope + "." + _name if scope else _name, _submod))
        # delete unused attributes
        for attr in unused_attributes:
            mod_itr, atoms = submod, attr.split(".")
            for atom in atoms[:-1]:
                mod_itr = getattr(mod_itr, atom)
            delattr(mod_itr, atoms[-1])

    for node in attr_nodes:
        # And (2): remove `get_attr` node from submod's arg list
        for user in copy.copy(node.users):
            assert user.op == "call_module"
            delete_user_reference(node, user)
        # And (3): remove the `get_attr` node from the root graph.
        split.graph.erase_node(node)

    split.delete_all_unused_submodules()
    split.graph.lint()
    split.recompile()
    return Pipe(split_gm=split, num_stages=part_idx + 1)


def split_module(
    module: torch.nn.Module,
    *,
    split_spec: dict[tuple[str, str], SplitPoint] | None = None,
    function: str | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    strict: bool = False,
) -> Pipe:
    if split_spec is not None:
        insert_stage_split_markers(module, split_spec=split_spec)
    return split_annotated_module(
        module,
        function=function,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
    )


def inter_stage_transfer(
    inter_stage_device_map: dict[int, int],
    stage_args: tuple[AnyTensor, ...] | None = None,
    stage_kwargs: dict[str, AnyTensor] | None = None,
) -> tuple[tuple[AnyTensor, ...], dict[str, AnyTensor]]:
    if len(inter_stage_device_map) == 1:
        pass
    else:
        raise NotImplementedError("TODO")


def insert_stage_arg_transfers(
    pipeline: Pipe,
    stage_device_map: tuple[tuple[int, ...]],
) -> tuple[torch.Graph, InterStageLiveSet]:
    """Insert tensor transfers at the start of each stage submodule."""
    pass


def construct_prepare_stage_args_function(
    pipeline: Pipe,
    stage_index: int,
    inter_stage_live_set: InterStageLiveSet,
) -> tuple[torch.Graph, InterStageLiveSet]:
    """Extract a graph that is a function preparing the arguments to execute a pipeline stage.

    Args:
        inter_stage_live_set: the set of arguments that is required to run the rest of the
            pipeline to completion. The initial set is the arguments to the pipeline.
    """
    pass


def _outline_submodules(orig_graph: torch.fx.Graph) -> torch.fx.GraphModule:
    # Create an empty GraphModule to hold the outlined modules
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: dict[str, torch.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {},
        module=new_module,
    ).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module
