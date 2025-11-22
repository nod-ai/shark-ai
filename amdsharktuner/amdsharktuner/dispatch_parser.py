# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func, iree_codegen, linalg  # type: ignore

from . import common


def get_parent_function_name(root_op: ir.Operation) -> str:
    """
    Returns the parent function's symbol name from a root operation.
    """
    # FIXME: This assumes the immediate parent is a function, but the root op
    # could be nested inside other operations (e.g., scf.if).
    func_op = root_op.parent.opview
    assert isinstance(func_op, func.FuncOp), f"Expected func.func, got {func_op.name}"
    return ir.StringAttr(func_op.name).value


def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
    mlir_module = None
    try:
        mlir_module = ir.Module.parse(mlir_text, ctx.mlir_ctx)
        ctx.logger.debug("MLIR parsing successful!")
    except ir.MLIRError as e:
        ctx.logger.error(f"Error parsing MLIR: {e}")
        raise RuntimeError(f"Error parsing MLIR: {e}")

    return mlir_module


@dataclass
class OpInfo:
    root_op: ir.Operation
    indexing_maps: list[ir.AffineMap]


@dataclass
class ContractionOpInfo(OpInfo):
    dims: common.ContractionDimensions
    matmul_size: common.ContractionSizes
    lhs_type: common.ShapedType
    rhs_type: common.ShapedType
    res_type: common.ShapedType


@dataclass
class ConvolutionOpInfo(OpInfo):
    dims: common.ContractionDimensions
    matmul_size: common.ContractionSizes
    lhs_type: common.ShapedType
    rhs_type: common.ShapedType
    res_type: common.ShapedType

    batch_sizes: list[int]
    output_image_sizes: list[int]
    output_channel_sizes: list[int]
    filter_loop_sizes: list[int]
    input_channel_sizes: list[int]
    depth_sizes: list[int]
    strides: list[int]
    dilations: list[int]

    # IGEMM details for TileAndFuse pipeline (None if not available).
    igemm_details: Optional[iree_codegen.IGEMMGenericConvDetails] = None


@dataclass
class AttentionOpInfo(OpInfo):
    """Information about an attention operation.

    Attention is decomposed into two matrix multiplications:
    - QK^T : Q @ K^T  (attention scores)
    - PV   : P @ V    (projected output after softmax)

    Assumed operand shapes: Q=[B,M,K1], K=[B,K2,K1], V=[B,K2,N], Res=[B,M,N]
    """

    # Dimension structure.
    domain_rank: int  # Total number of dimensions in the operation.
    batch_dims: list[int]  # Indices of batch dimensions.
    m_dims: list[int]  # Indices of M dimensions (query sequence length).
    n_dims: list[int]  # Indices of N dimensions (output/value dimensions).
    k1_dims: list[int]  # Indices of K1 dimensions (query/key feature dimensions).
    k2_dims: list[int]  # Indices of K2 dimensions (key sequence length).

    # Dimension sizes.
    batch_sizes: list[int]  # Sizes of batch dimensions.
    m_sizes: list[int]  # Sizes of M dimensions.
    n_sizes: list[int]  # Sizes of N dimensions.
    k1_sizes: list[int]  # Sizes of K1 dimensions.
    k2_sizes: list[int]  # Sizes of K2 dimensions.

    # Operand types.
    query_type: ir.Type  # MLIR type of query tensor.
    key_type: ir.Type  # MLIR type of key tensor.
    value_type: ir.Type  # MLIR type of value tensor.
    output_type: ir.Type  # MLIR type of output tensor.

    # Transpose flags (True if dimension is not in the expected position).
    transposed_q: bool  # True if Q is logically transposed (k1 dim is not last).
    transposed_k: bool  # True if K is logically transposed (k1 dim is not last).
    transposed_v: bool  # True if V is logically transposed (k2 dim is not last).

    # Decomposed matmul metadata.
    qk_matmul: common.MatmulShapeType  # Shape metadata for Q @ K^T matmul.
    pv_matmul: common.MatmulShapeType  # Shape metadata for P @ V matmul.


class DispatchParser(metaclass=ABCMeta):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        self._root_op = root_op
        self._tuner_ctx = tuner_ctx
        self._op_info: Optional[OpInfo] = None

    def get_root_op(self) -> ir.Operation:
        return self._root_op

    def get_iter_dim_size(
        self, iter_dim: int, operand_idx: int, indexing_maps: list[ir.AffineMap]
    ) -> int:
        root_op = self.get_root_op()
        operand_type = root_op.operands[operand_idx].type
        indexing_map = indexing_maps[operand_idx]
        tensor_dim = list(indexing_map.results).index(ir.AffineExpr.get_dim(iter_dim))
        return operand_type.shape[tensor_dim]

    @abstractmethod
    def has_valid_root_op(self) -> bool:
        """Check if the root_op is valid and supported by this tuner."""
        pass

    @abstractmethod
    def get_op_info(self) -> OpInfo:
        """Extract and return OpInfo for this operation."""
        pass


class ContractionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)
        root_op = self.get_root_op()
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        assert contraction_dims, "no contraction dimensions"
        dims = common.ContractionDimensions(
            batch=list(contraction_dims.batch),
            m=list(contraction_dims.m),
            n=list(contraction_dims.n),
            k=list(contraction_dims.k),
        )
        res_maps = linalg.get_indexing_maps(root_op)
        indexing_maps = [map_attr.value for map_attr in res_maps]

        lhs_dims = common.get_map_result_dim_positions(indexing_maps[0])
        rhs_dims = common.get_map_result_dim_positions(indexing_maps[1])
        res_dims = common.get_map_result_dim_positions(indexing_maps[2])

        assert lhs_dims, "no lhs dimensions"
        assert rhs_dims, "no rhs dimensions"
        assert res_dims, "no result dimensions"

        lhs_type = ir.RankedTensorType(root_op.operands[0].type)
        rhs_type = ir.RankedTensorType(root_op.operands[1].type)
        res_type = ir.RankedTensorType(root_op.operands[2].type)

        matmul_size = common.ContractionSizes(
            M=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.m],
            N=[rhs_type.shape[rhs_dims.index(dim)] for dim in contraction_dims.n],
            K=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.k],
            B=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch],
        )

        self._op_info: ContractionOpInfo = ContractionOpInfo(
            root_op=root_op,
            indexing_maps=indexing_maps,
            dims=dims,
            matmul_size=matmul_size,
            lhs_type=common.ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=common.ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=common.ShapedType(res_type.shape, res_type.element_type),
        )

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return linalg.isa_contraction_op(root_op)

    def get_op_info(self) -> ContractionOpInfo:
        return self._op_info


class ConvolutionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)
        root_op = self.get_root_op()
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        assert convolution_dims, "no convolution dimensions"

        batch_indices = list(convolution_dims.batch)
        output_image_indices = list(convolution_dims.output_image)
        output_channel_indices = list(convolution_dims.output_channel)
        filter_loop_indices = list(convolution_dims.filter_loop)
        input_channel_indices = list(convolution_dims.input_channel)
        depth_indices = list(convolution_dims.depth)
        strides = list(convolution_dims.strides)
        dilations = list(convolution_dims.dilations)

        res_maps = linalg.get_indexing_maps(root_op)
        indexing_maps = [map_attr.value for map_attr in res_maps]

        contraction_dims = common.ContractionDimensions(
            batch=depth_indices,
            m=batch_indices + output_image_indices,
            n=output_channel_indices,
            k=filter_loop_indices + input_channel_indices,
        )

        batch_sizes = (
            [self.get_iter_dim_size(d, 2, indexing_maps) for d in batch_indices]
            if batch_indices
            else []
        )
        output_image_sizes = (
            [self.get_iter_dim_size(d, 2, indexing_maps) for d in output_image_indices]
            if output_image_indices
            else []
        )
        output_channel_sizes = (
            [
                self.get_iter_dim_size(d, 2, indexing_maps)
                for d in output_channel_indices
            ]
            if output_channel_indices
            else []
        )
        filter_loop_sizes = (
            [self.get_iter_dim_size(d, 1, indexing_maps) for d in filter_loop_indices]
            if filter_loop_indices
            else []
        )
        input_channel_sizes = (
            [self.get_iter_dim_size(d, 0, indexing_maps) for d in input_channel_indices]
            if input_channel_indices
            else []
        )
        depth_sizes = (
            [self.get_iter_dim_size(d, 2, indexing_maps) for d in depth_indices]
            if depth_indices
            else []
        )

        matmul_size = common.ContractionSizes(
            B=depth_sizes,
            M=batch_sizes + output_image_sizes,
            N=output_channel_sizes,
            K=filter_loop_sizes + input_channel_sizes,
        )

        lhs_type = root_op.operands[0].type
        rhs_type = root_op.operands[1].type
        res_type = root_op.operands[2].type

        # Get IGEMM details for potential use with the TileAndFuse pipeline.
        # This provides flattened K dimensions and proper M/N/K categorization
        # for any convolution layout (nhwc_hwcf, nchw_fchw, etc.).
        igemm_details = iree_codegen.get_igemm_generic_conv_details(root_op)

        self._op_info: ConvolutionOpInfo = ConvolutionOpInfo(
            root_op=root_op,
            indexing_maps=indexing_maps,
            dims=contraction_dims,
            matmul_size=matmul_size,
            lhs_type=common.ShapedType(lhs_type.shape, lhs_type.element_type),
            rhs_type=common.ShapedType(rhs_type.shape, rhs_type.element_type),
            res_type=common.ShapedType(res_type.shape, res_type.element_type),
            batch_sizes=batch_sizes,
            output_image_sizes=output_image_sizes,
            output_channel_sizes=output_channel_sizes,
            filter_loop_sizes=filter_loop_sizes,
            input_channel_sizes=input_channel_sizes,
            depth_sizes=depth_sizes,
            strides=strides,
            dilations=dilations,
            igemm_details=igemm_details,
        )

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        if not linalg.isa_convolution_op(root_op):
            return False
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        assert convolution_dims, "no convolution dimensions"
        # Only allow 'nhwc_hwcf' convs.
        # TODO: This dispatch parser class supports more layouts, but constraint
        #       generation is not tested. Relax this check as support is verified.
        if (
            list(convolution_dims.batch) != [0]
            or list(convolution_dims.output_image) != [1, 2]
            or list(convolution_dims.output_channel) != [3]
            or list(convolution_dims.filter_loop) != [4, 5]
            or list(convolution_dims.input_channel) != [6]
            or list(convolution_dims.depth) != []
        ):
            return False
        return True

    def get_op_info(self) -> ConvolutionOpInfo:
        return self._op_info


class AttentionOpInterfaceParser(DispatchParser):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

        root_op = self.get_root_op()
        indexing_maps_attr = root_op.attributes["indexing_maps"]
        indexing_maps = [attr.value for attr in indexing_maps_attr]
        q_map = indexing_maps[0]
        k_map = indexing_maps[1]
        v_map = indexing_maps[2]
        o_map = indexing_maps[-1]

        raw_opinfo = iree_codegen.get_attention_op_detail(q_map, k_map, v_map, o_map)
        assert raw_opinfo, "no attention info"

        batch_indices = list(raw_opinfo.batch_dims)
        m_indices = list(raw_opinfo.m_dims)
        n_indices = list(raw_opinfo.n_dims)
        k1_indices = list(raw_opinfo.k1_dims)
        k2_indices = list(raw_opinfo.k2_dims)

        q_type = ir.RankedTensorType(root_op.operands[0].type)
        k_type = ir.RankedTensorType(root_op.operands[1].type)
        v_type = ir.RankedTensorType(root_op.operands[2].type)
        output_type = ir.RankedTensorType(root_op.results[0].type)

        q_shape = q_type.shape
        k_shape = k_type.shape
        v_shape = v_type.shape

        batch_sizes = (
            [self.get_iter_dim_size(d, 0, indexing_maps) for d in batch_indices]
            if batch_indices
            else []
        )
        m_sizes = (
            [self.get_iter_dim_size(d, 0, indexing_maps) for d in m_indices]
            if m_indices
            else []
        )
        n_sizes = (
            [self.get_iter_dim_size(d, 2, indexing_maps) for d in n_indices]
            if n_indices
            else []
        )
        k1_sizes = (
            [self.get_iter_dim_size(d, 0, indexing_maps) for d in k1_indices]
            if k1_indices
            else []
        )
        k2_sizes = (
            [self.get_iter_dim_size(d, 1, indexing_maps) for d in k2_indices]
            if k2_indices
            else []
        )

        mDim = raw_opinfo.m_dims[-1]
        k1Dim = raw_opinfo.k1_dims[-1]
        k2Dim = raw_opinfo.k2_dims[-1]
        nDim = raw_opinfo.n_dims[-1]

        q_last_expr = q_map.results[-1]
        k_last_expr = k_map.results[-1]
        v_last_expr = v_map.results[-1]

        q_dim_expr = ir.AffineDimExpr(q_last_expr)
        k_dim_expr = ir.AffineDimExpr(k_last_expr)
        v_dim_expr = ir.AffineDimExpr(v_last_expr)

        transposed_k = k1Dim != k_dim_expr.position
        transposed_v = k2Dim != v_dim_expr.position
        transposed_q = k1Dim != q_dim_expr.position

        q_dims = common.get_map_result_dim_positions(q_map)
        k_dims = common.get_map_result_dim_positions(k_map)
        v_dims = common.get_map_result_dim_positions(v_map)

        assert q_dims, "no query dims from attention op"
        assert k_dims, "no key dims from attention op"
        assert v_dims, "no value dims from attention op"

        f32_type = ir.F32Type.get()

        qk_matmul = common.MatmulShapeType(
            m=q_shape[q_dims.index(mDim)],
            n=k_shape[k_dims.index(k2Dim)],
            k=q_shape[q_dims.index(k1Dim)],
            lhs_type=q_type.element_type,
            rhs_type=k_type.element_type,
            acc_type=f32_type,
        )

        pv_matmul = common.MatmulShapeType(
            m=q_shape[q_dims.index(mDim)],
            n=v_shape[v_dims.index(nDim)],
            k=v_shape[v_dims.index(k2Dim)],
            lhs_type=v_type.element_type,
            rhs_type=v_type.element_type,
            acc_type=output_type.element_type,
        )

        self._op_info: AttentionOpInfo = AttentionOpInfo(
            root_op=root_op,
            indexing_maps=indexing_maps,
            domain_rank=raw_opinfo.domain_rank,
            batch_dims=batch_indices,
            m_dims=m_indices,
            n_dims=n_indices,
            k1_dims=k1_indices,
            k2_dims=k2_indices,
            batch_sizes=batch_sizes,
            m_sizes=m_sizes,
            n_sizes=n_sizes,
            k1_sizes=k1_sizes,
            k2_sizes=k2_sizes,
            query_type=q_type.element_type,
            key_type=k_type.element_type,
            value_type=v_type.element_type,
            output_type=output_type.element_type,
            transposed_q=transposed_q,
            transposed_k=transposed_k,
            transposed_v=transposed_v,
            qk_matmul=qk_matmul,
            pv_matmul=pv_matmul,
        )

    def has_valid_root_op(self) -> bool:
        root_op = self.get_root_op()
        return iree_codegen.isa_attention_op(root_op)

    def get_op_info(self) -> AttentionOpInfo:
        return self._op_info
