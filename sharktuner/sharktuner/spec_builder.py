# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABC, abstractmethod

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, preprocessing_transform, transform  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *

ROOT_OP_ATTR_NAME = "root_op"


def get_matcher_named_sequence_name(root_op: ir.Operation) -> str:
    """
    Returns the symbol name for a transform dialect named sequence matcher.
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


class ConvolutionSpecBuilder(SpecBuilder):
    def __init__(self, op_info: ConvolutionOpInfo):
        super().__init__(op_info)
        self.op_info: ConvolutionOpInfo = op_info

    def build_matcher(
        self,
        entry_block: ir.Block,
        body_target: ir.Value,
        config_list: list[common.TuningConfiguration],
    ) -> tuple[ir.Value, list[ir.Value]]:
        """
        Gets a convolution matcher using transform.iree.match.convolution.
        """
        lhs_elem_type = self.op_info.lhs_type.element_type
        rhs_elem_type = self.op_info.rhs_type.element_type
        res_elem_type = self.op_info.res_type.element_type

        with ir.InsertionPoint(entry_block):
            (
                batch,
                out_img,
                out_ch,
                filt,
                in_ch,
                depth,
                strides,
                dilations,
            ) = preprocessing_transform.MatchConvolutionOp(
                operand_handle=body_target,
                lhs_type=lhs_elem_type,
                rhs_type=rhs_elem_type,
                output_type=res_elem_type,
                indexing_maps=self.op_info.indexing_maps,
            )

            preprocessing_transform.MatchDimsEqualOp(batch, self.op_info.batch_sizes)
            preprocessing_transform.MatchDimsEqualOp(
                out_img, self.op_info.output_image_sizes
            )
            preprocessing_transform.MatchDimsEqualOp(
                out_ch, self.op_info.output_channel_sizes
            )
            preprocessing_transform.MatchDimsEqualOp(
                filt, self.op_info.filter_loop_sizes
            )
            preprocessing_transform.MatchDimsEqualOp(
                in_ch, self.op_info.input_channel_sizes
            )
            preprocessing_transform.MatchDimsEqualOp(depth, self.op_info.depth_sizes)
            preprocessing_transform.MatchDimsEqualOp(strides, self.op_info.strides)
            preprocessing_transform.MatchDimsEqualOp(dilations, self.op_info.dilations)

            config_params = self.create_config_params(config_list)
            return body_target, config_params


class AttentionSpecBuilder(SpecBuilder):
    def __init__(self, op_info: AttentionOpInfo):
        super().__init__(op_info)
        self.op_info: AttentionOpInfo = op_info

    def build_matcher(
        self,
        entry_block: ir.Block,
        body_target: ir.Value,
        config_list: list[common.TuningConfiguration],
    ) -> tuple[ir.Value, list[ir.Value]]:
        """Gets an attention matcher using transform.iree.match.attention."""

        query_elem_type = self.op_info.query_type
        key_elem_type = self.op_info.key_type
        value_elem_type = self.op_info.value_type
        output_elem_type = self.op_info.output_type

        batch_sizes = self.op_info.batch_sizes
        m_sizes = self.op_info.m_sizes
        n_sizes = self.op_info.n_sizes
        k1_sizes = self.op_info.k1_sizes
        k2_sizes = self.op_info.k2_sizes

        with ir.InsertionPoint(entry_block):
            batch, m, n, k1, k2 = preprocessing_transform.MatchAttentionOp(
                operand_handle=body_target,
                query_type=query_elem_type,
                key_type=key_elem_type,
                value_type=value_elem_type,
                output_type=output_elem_type,
                indexing_maps=self.op_info.indexing_maps,
            )

            preprocessing_transform.MatchDimsEqualOp(batch, batch_sizes)
            preprocessing_transform.MatchDimsEqualOp(m, m_sizes)
            preprocessing_transform.MatchDimsEqualOp(n, n_sizes)
            preprocessing_transform.MatchDimsEqualOp(k1, k1_sizes)
            preprocessing_transform.MatchDimsEqualOp(k2, k2_sizes)

            config_params = self.create_config_params(config_list)
            return body_target, config_params
