# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABC, abstractmethod

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore
from iree.compiler.dialects import preprocessing_transform  # type: ignore

from .common import *
from .dispatch_constraints import *
from .dispatch_parser import *

ROOT_OP_ATTR_NAME = "root_op"


class SpecBuilder(ABC):
    def __init__(self, opinfo: OpInfo):
        self.opinfo = opinfo

    def create_config_params(
        self, config_list: list[common.TuningConfiguration]
    ) -> list[ir.Value]:
        """
        Creates parameter constant operations for each configuration.
        """
        config_params = []
        for config in config_list:
            config_param = transform.ParamConstantOp(
                transform.AnyParamType.get(),
                config.configuration,
            ).result
            config_params.append(config_param)
        return config_params

    @staticmethod
    def get_placeholder_spec(context: ir.Context) -> ir.Module:
        """
        Creates a placeholder Transform Dialect spec that does nothing.

        This is used for the baseline (index 0) configuration where no
        tuning spec is applied. It simply yields the input variant operation
        without any modifications.

        """
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            module.operation.attributes[
                "transform.with_named_sequence"
            ] = ir.UnitAttr.get()

            with ir.InsertionPoint(module.body):
                input_types = [transform.AnyOpType.get()]
                output_types = [transform.AnyOpType.get()]

                arg_attrs = [
                    {"transform.readonly": ir.UnitAttr.get()} for _ in input_types
                ]

                named_seq = transform.NamedSequenceOp(
                    "__kernel_config",
                    input_types,
                    output_types,
                    arg_attrs=arg_attrs,
                )

                named_seq.operation.attributes[
                    "iree_codegen.tuning_spec_entrypoint"
                ] = ir.UnitAttr.get()

                with ir.InsertionPoint(named_seq.body):
                    variant_op = named_seq.bodyTarget
                    transform.YieldOp([variant_op])

            return module

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
            self.opinfo.func_name,
            input_types,
            output_types,
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
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

        arg_attrs = [{"transform.readonly": ir.UnitAttr.get()} for _ in input_types]

        named_seq = transform.NamedSequenceOp(
            "apply_op_config",
            input_types,
            output_types,
            arg_attrs=arg_attrs,
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
            arg_attrs=[{"transform.consumed": ir.UnitAttr.get()}],
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
                [self.opinfo.func_name],
                ["apply_op_config"],
            ).updated

            transform.YieldOp([result])

        return named_seq

    def build_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        """
        Builds a Transform Dialect spec module using Python bindings.
        """
        context = self.opinfo.context
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
    def __init__(self, opinfo: ContractionOpInfo):
        super().__init__(opinfo)
        self.opinfo: ContractionOpInfo = opinfo

    def build_matcher(
        self,
        entry_block: ir.Block,
        body_target: ir.Value,
        config_list: list[common.TuningConfiguration],
    ) -> tuple[ir.Value, list[ir.Value]]:
        """
        Gets a contraction matcher using transform.iree.match.contraction.
        """
        lhs_elem_type = self.opinfo.lhs_type.element_type
        rhs_elem_type = self.opinfo.rhs_type.element_type
        res_elem_type = self.opinfo.res_type.element_type

        m_dims = self.opinfo.matmul_size.M
        n_dims = self.opinfo.matmul_size.N
        k_dims = self.opinfo.matmul_size.K
        batch_dims = self.opinfo.matmul_size.B

        with ir.InsertionPoint(entry_block):
            batch, m, n, k = preprocessing_transform.MatchContractionOp(
                operand_handle=body_target,
                lhs_type=lhs_elem_type,
                rhs_type=rhs_elem_type,
                output_type=res_elem_type,
                indexing_maps=self.opinfo.indexing_maps,
            )

            preprocessing_transform.MatchDimsEqualOp(batch, batch_dims)
            preprocessing_transform.MatchDimsEqualOp(m, m_dims)
            preprocessing_transform.MatchDimsEqualOp(n, n_dims)
            preprocessing_transform.MatchDimsEqualOp(k, k_dims)

            config_params = self.create_config_params(config_list)
            return body_target, config_params


class ConvolutionSpecBuilder(SpecBuilder):
    def __init__(self, opinfo: ConvolutionOpInfo):
        super().__init__(opinfo)
        self.opinfo: ConvolutionOpInfo = opinfo

    def build_matcher(
        self,
        entry_block: ir.Block,
        body_target: ir.Value,
        config_list: list[common.TuningConfiguration],
    ) -> tuple[ir.Value, list[ir.Value]]:
        """
        Gets a convolution matcher using transform.iree.match.convolution.
        """
        lhs_elem_type = self.opinfo.lhs_type.element_type
        rhs_elem_type = self.opinfo.rhs_type.element_type
        res_elem_type = self.opinfo.res_type.element_type

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
                indexing_maps=self.opinfo.indexing_maps,
            )

            preprocessing_transform.MatchDimsEqualOp(batch, self.opinfo.batch)
            preprocessing_transform.MatchDimsEqualOp(out_img, self.opinfo.output_image)
            preprocessing_transform.MatchDimsEqualOp(out_ch, self.opinfo.output_channel)
            preprocessing_transform.MatchDimsEqualOp(filt, self.opinfo.filter_loop)
            preprocessing_transform.MatchDimsEqualOp(in_ch, self.opinfo.input_channel)
            preprocessing_transform.MatchDimsEqualOp(depth, self.opinfo.depth)
            preprocessing_transform.MatchDimsEqualOp(strides, self.opinfo.strides)
            preprocessing_transform.MatchDimsEqualOp(dilations, self.opinfo.dilations)

            config_params = self.create_config_params(config_list)
            return body_target, config_params


class AttentionSpecBuilder(SpecBuilder):
    def __init__(self, opinfo: AttentionOpInfo):
        super().__init__(opinfo)
        self.opinfo: AttentionOpInfo = opinfo

    def build_matcher(
        self,
        entry_block: ir.Block,
        body_target: ir.Value,
        config_list: list[common.TuningConfiguration],
    ) -> tuple[ir.Value, list[ir.Value]]:
        """Gets an attention matcher using transform.iree.match.attention."""

        query_elem_type = self.opinfo.query_type
        key_elem_type = self.opinfo.key_type
        value_elem_type = self.opinfo.value_type
        output_elem_type = self.opinfo.output_type

        batch_dims = self.opinfo.batch_sizes
        m_dims = self.opinfo.M
        n_dims = self.opinfo.N
        k1_dims = self.opinfo.K1
        k2_dims = self.opinfo.K2

        with ir.InsertionPoint(entry_block):
            batch, m, n, k1, k2 = preprocessing_transform.MatchAttentionOp(
                operand_handle=body_target,
                query_type=query_elem_type,
                key_type=key_elem_type,
                value_type=value_elem_type,
                output_type=output_elem_type,
                indexing_maps=self.opinfo.indexing_maps,
            )

            preprocessing_transform.MatchDimsEqualOp(batch, batch_dims)
            preprocessing_transform.MatchDimsEqualOp(m, m_dims)
            preprocessing_transform.MatchDimsEqualOp(n, n_dims)
            preprocessing_transform.MatchDimsEqualOp(k1, k1_dims)
            preprocessing_transform.MatchDimsEqualOp(k2, k2_dims)

            config_params = self.create_config_params(config_list)
            return body_target, config_params
