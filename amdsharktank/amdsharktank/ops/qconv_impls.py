# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implementations for op variants that are fully quantized.
"""

from typing import Optional

import warnings

import torch
import torch.nn.functional as F

from amdsharktank import kernels

from amdsharktank.types import (
    AnyTensor,
    QuantizedTensor,
    PlanarQuantizedTensor,
    TensorScaledLayout,
)
from amdsharktank.utils import debugging

from .signatures import (
    IntOrSequenceInt,
    conv2d,
    elementwise,
)


def qconv2d_tensor_scaled(
    input: QuantizedTensor,
    weight: QuantizedTensor,
    bias: Optional[AnyTensor] = None,
    *,
    stride: IntOrSequenceInt = 1,
    padding: IntOrSequenceInt = 0,
    dilation: IntOrSequenceInt = 1,
    groups: IntOrSequenceInt = 1,
    accum_dtype: torch.dtype,
):
    # Grouped conv not yet supported.
    if groups != 1:
        return NotImplemented

    # Only handle tensor scaled layouts.
    if not issubclass(input.layout_type, TensorScaledLayout) or not issubclass(
        weight.layout_type, TensorScaledLayout
    ):
        return NotImplemented

    # Now we know that both the x/weight are TensorScaledLayout. There are still
    # degrees of freedom:
    #  * Either/both can be per-tensor or per-axis scaled (d is 0D or d is nd>0).
    #  * Either/both can have offsets of not (m is not None).
    input_layout: TensorScaledLayout = input.unpack()
    weight_layout: TensorScaledLayout = weight.unpack()

    # # Handle integer and fp8 quantizations.
    if (
        input_layout.qs.dtype.is_floating_point
        or weight_layout.qs.dtype.is_floating_point
    ):
        if (
            input_layout.qs.dtype != torch.float8_e4m3fnuz
            or weight_layout.qs.dtype != torch.float8_e4m3fnuz
        ):
            return NotImplemented

    # Bias is both optional and may either be quantized or fp.
    bias_qs = None
    rescale_d = None
    if bias is not None:
        if isinstance(bias, QuantizedTensor):
            bias_layout: TensorScaledLayout = bias.unpack()
            if isinstance(bias_layout, TensorScaledLayout):
                bias_qs = bias_layout.qs
                # If a quantized bias is provided, use its scale as the output scale and
                # add directly in integer. A quantized bias cannot be arbitrary and must
                # be a symmetric quantization of the output scale. This is not verified
                # and driven by the data.
                # Broadcast the bias scale to the channels in the NCHW output.
                rescale_d = bias_layout.d.reshape(1, -1, 1, 1)
            else:
                warnings.warn(f"unsupported qconv bias quantization: {bias_layout}")

    # Alias components (d=scale, qs=quantized samples, m=offset).
    if accum_dtype is None:
        accum_dtype = torch.int32
        if weight_layout.qs.dtype.is_floating_point:
            accum_dtype = torch.float32
    input_d = input_layout.d
    input_dtype = input_layout.dtype
    input_qs = input_layout.qs
    input_m = input_layout.m
    weight_d = weight_layout.d
    weight_qs = weight_layout.qs
    weight_m = weight_layout.m

    # Verify that quantization axis meets our requirements.
    flat_input_d, flat_input_m = _flatten_input_scale_offset_channels(input_d, input_m)
    flat_weight_d, flat_weight_m = _flatten_weight_scale_offset_channels(
        weight_d, weight_m
    )
    if flat_input_d is None or flat_weight_d is None:
        return NotImplemented

    # No quantized bias: infer the output rescale.
    if rescale_d is None:
        # Output scale by the product of input and weight scale and broadcast
        # to the output NCHW shape.
        rescale_d = (flat_input_d * flat_weight_d).reshape(1, -1, 1, 1)

    # Perform the actual convolution.
    stride = _expand_int_to_2_tuple(stride)
    padding = _expand_int_to_2_tuple(padding)
    dilation = _expand_int_to_2_tuple(dilation)
    extended_padding_list = [item for item in padding for _ in range(2)]
    padded_input = F.pad(input_qs, pad=extended_padding_list)
    y_qs = _invoke_conv2d_kernel(
        padded_input,
        weight_qs,
        bias_qs.to(accum_dtype) if bias_qs is not None else None,
        stride,
        dilation,
        accum_dtype=accum_dtype,
    )

    # Apply offset corrections.
    if input_m is not None:
        # Apply offset correction for asymmetric input.
        # At the time of this writing, this was not a common case.
        # Apply the offset fix to the channel output axis (1).
        input_offset_fix = torch.sum(
            torch.flatten(weight_qs, 1), dim=1, dtype=accum_dtype
        )
        input_offset_fix = input_offset_fix * flat_input_m
        input_offset_fix = input_offset_fix.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_qs = y_qs - input_offset_fix
    if flat_weight_m is not None:
        # Apply offset correction for asymmetric weight.
        # At the time of this writing, this was the common case.
        # The weight_m shape is the offset relative to weight and needs to
        # be reshaped to broadcast to the NCHW output. Since all but the
        # output channels are zero, just
        # Note that we sum first to reduce the dimensionality by channel
        # prior, reducing memory and total computation.
        weight_offset_fix = torch.sum(
            padded_input, dim=1, keepdim=True, dtype=accum_dtype
        )
        weight_offset_fix = _invoke_pooling_sum_kernel(
            weight_offset_fix,
            [weight_qs.shape[2], weight_qs.shape[3]],
            stride,
            dilation,
            accum_dtype=accum_dtype,
        )
        weight_offset_fix = weight_offset_fix * flat_weight_m.unsqueeze(0).unsqueeze(
            2
        ).unsqueeze(3)
        y_qs = y_qs - weight_offset_fix
    if input_m is not None and weight_m is not None:
        # Apply joint offset correction if both input and weight are asymmetric.
        # At the time of this writing, this was not a common case.
        joint_fix = (
            flat_input_m.unsqueeze(0)
            * flat_weight_m.unsqueeze(0)
            * torch.tensor(
                weight_qs.shape[1] * weight_qs.shape[2] * weight_qs.shape[3],
                dtype=accum_dtype,
            )
        )
        joint_fix = joint_fix.unsqueeze(2).unsqueeze(3)
        y_qs = y_qs + joint_fix

    output_shape = list(y_qs.shape)
    y = PlanarQuantizedTensor(
        shape=output_shape,
        layout=TensorScaledLayout(
            shape=output_shape,
            d=rescale_d,
            qs=y_qs,
            dtype=input_dtype,
        ),
    )

    # If we have an unquantized bias, dequantize the result and add here.
    if bias is not None and bias_qs is None:
        y = y.unpack().dequant()
        y = elementwise(torch.add, y, bias.reshape(-1, 1, 1))
    return y


conv2d.override(QuantizedTensor, QuantizedTensor)(qconv2d_tensor_scaled)
conv2d.override(QuantizedTensor, QuantizedTensor, AnyTensor)(qconv2d_tensor_scaled)


def _invoke_conv2d_kernel(input, weight, bias, stride, dilation, *, accum_dtype):
    """Does a low level invocation of a conv2d integer kernel on an explicitly padded input.

    This presumes that the stride/padding/dilation have already been normalized
    to 2-tuples.

    It is advantageous in some situations to use fp emulation of an int kernel
    so we fork here.
    """
    if debugging.flags.use_custom_iree_kernels:
        # True int kernel.
        if bias is None:
            # We don't have any non-test use of convs without bias, so just
            # use a zero vector for the None case. If this becomes common, we
            # may want to support an optional bias through the kernel.
            bias = torch.zeros(
                (weight.shape[1],), dtype=accum_dtype, device=input.device
            )
        y_qs = kernels.conv_2d_nchw_fchw(
            input,
            weight,
            bias,
            stride,
            dilation,
            output_dtype=str(accum_dtype),
        )
    else:
        # FP emulation.
        y_qs = torch.nn.functional.conv2d(
            input.to(dtype=torch.float32),
            weight.to(dtype=torch.float32),
            bias=bias.to(dtype=torch.float32) if bias is not None else None,
            stride=stride,
            dilation=dilation,
        ).to(dtype=accum_dtype)

    return y_qs


def _invoke_pooling_sum_kernel(input, kernel_size, stride, dilation, *, accum_dtype):
    """Invokes either a custom integer pooling sum or the built-in fp avg_pool2d
    kernel on an explicitly padded input.
    """
    if debugging.flags.use_custom_iree_kernels:
        output = kernels.pooling_nchw_sum(
            input,
            kernel_size,
            stride,
            dilation,
        )
    else:
        output = torch.nn.functional.avg_pool2d(
            input.to(dtype=torch.float32),
            kernel_size,
            stride=stride,
            divisor_override=1,
        ).to(dtype=accum_dtype)
    return output


def _flatten_input_scale_offset_channels(d, m):
    """Flattens either a 4d or 0d scale/offset as [N, C, H, W] to 1D.

    Returns None, None if not scaled along the output channel dimension.
    """
    d_rank = len(d.shape)
    assert m is None or d_rank == len(m.shape), "Mismatched d/m ranks"
    if d_rank == 0:
        return d.unsqueeze(0), m.unsqueeze(0) if m is not None else None
    elif d_rank != 4:
        return None, None

    # Flatten d.
    d_x, d_c, d_y, d_z = d.shape
    if d_x != 1 or d_y != 1 or d_z != 1:
        return None, None
    d = d.squeeze()

    # Flatten m.
    if m is not None:
        m_x, m_c, m_y, m_z = m.shape
        if m_x != 1 or m_y != 1 or m_z != 1:
            return None, None
        m = m.squeeze()
    return d, m


def _flatten_weight_scale_offset_channels(d, m):
    """Flattens either a 4d or 0d scale/offset as [C, 1, 1, 1] to 1D.

    Returns None, None if not scaled along the output channel dimension.
    """
    d_rank = len(d.shape)
    assert m is None or d_rank == len(m.shape), "Mismatched d/m ranks"
    if d_rank == 0:
        return d.unsqueeze(0), m.unsqueeze(0) if m is not None else None
    elif d_rank != 4:
        return None, None

    # Flatten d.
    d_c, d_x, d_y, d_z = d.shape
    if d_x != 1 or d_y != 1 or d_z != 1:
        return None, None
    d = d.squeeze()

    # Flatten m.
    if m is not None:
        m_c, m_x, m_y, m_z = m.shape
        if m_x != 1 or m_y != 1 or m_z != 1:
            return None, None
        m = m.squeeze()
    return d, m


def _expand_int_to_2_tuple(int_or_list) -> tuple[int, int]:
    if isinstance(int_or_list, int):
        return (int_or_list, int_or_list)
    assert len(int_or_list) == 2, "Expected int or (i, i) sequence"
    return int_or_list
