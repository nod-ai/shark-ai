# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Quantizer Tensors
These tensors contain quantization parameters that can be used to quantize some other
tensor. These are typically stored in a dataset to signal a transformation into
a quantized representation for some layer (typically for activations or other dynamic
value) for which the underlying parameters themselves are fixed.

Note that there is no need for a "DequantizerTensor" or a "dequantize" method on
this class, since any `QuantizedTensor` already knows how to dequantize itself.
"""

from typing import Any, List, Optional, Tuple

from abc import abstractmethod

import torch

from sharktank.utils.io import ShardedArchiveBuilder

from .layouts import (
    BlockScaledFp4Layout,
    TensorScaledLayout,
)

from .layout_utils import (
    pack_fp4_e2m1_to_uint8,
    saturate_cast,
)

from .ocp_floats import (
    compute_fp4_block_scales,
    float32_to_fp4_e2m1,
    e8m0_to_float32,
    float32_to_e8m0,
)

from .tensors import (
    InferenceTensor,
    InferenceTensorMetadata,
    PlanarQuantizedTensor,
    PrimitiveTensor,
    QuantizedTensor,
    UnnamedTensorName,
    register_inference_tensor,
    serialized_name_to_dtype,
    dtype_to_serialized_name,
)

__all__ = [
    "DynamicFp4BlockQuantizer",
    "DynamicScaledQuantizer",
    "QuantizerTensor",
    "StaticFp4BlockQuantizer",
    "StaticScaledQuantizer",
]


class QuantizerTensor(InferenceTensor):
    """A tensor that knows how to quantize some other tensor."""

    def quantize(
        self, t: torch.Tensor | InferenceTensor, *, name: str = UnnamedTensorName
    ) -> QuantizedTensor:
        """Quantize from an arbitrary source tensor (framework or inference).

        This has some additional heuristics for unpacking and rescaling
        of InferenceTensors.
        """
        if isinstance(t, InferenceTensor):
            if isinstance(t, PrimitiveTensor):
                raw_tensor = t.as_torch()
            elif isinstance(t, QuantizedTensor):
                import warnings

                warnings.warn(f"Requantizing already quantized tensor {t} to {self}")
                raw_tensor = t.unpack().dequant()
            else:
                raise TypeError(
                    f"Unsupported tensor type in QuantizerTensor.quantize: {type(t)}"
                )
        else:
            assert isinstance(t, torch.Tensor)
            raw_tensor = t
        return self._quantize_raw_tensor(raw_tensor, name=name)

    @abstractmethod
    def _quantize_raw_tensor(self, t: torch.Tensor, *, name: str) -> QuantizedTensor:
        """Performs a quantizing transformation on t, returning a QuantizeTensor."""
        ...


@register_inference_tensor
class StaticScaledQuantizer(QuantizerTensor):
    """Quantizes to a `TensorScaledLayout` (per-tensor) or (TBD) for per-axis.

    If `scale` is a scalar, it produces a PlanarQuantizedTensor of a
    TensorScaledLayout where the `d` (scale) is the reciprocal of the scale
    specified here.

    An optional pre-scaled `offset` can be provided that:

    * Quantizing: Will be added to the scaled value prior to rounding/clamping.
    * Dequantizing: Will be subtracted from the quantized value prior to
      scaling.

    If provided, the offset must be of the specified target `dtype`.
    """

    def __init__(
        self,
        *,
        scale: torch.Tensor,
        dtype: torch.dtype,
        axis: Optional[int] = None,
        reciprocal_scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        disable_saturate: bool = False,
        name: str = UnnamedTensorName,
    ):
        super().__init__(shape=scale.shape, name=name)
        self._axis, (
            self._scale,
            self._reciprocal_scale,
            self._offset,
        ) = _norm_per_axis_param(axis, scale, reciprocal_scale, offset)
        if self._reciprocal_scale is None:
            self._reciprocal_scale = 1.0 / self._scale
        self._dtype = dtype
        self._disable_saturate = disable_saturate
        assert self._scale.shape == self._reciprocal_scale.shape
        assert self._scale.dtype == self._reciprocal_scale.dtype
        if self._offset is not None:
            assert self._offset.shape == self._scale.shape
            assert self._offset.dtype == dtype
        if self._axis is not None:
            assert len(self._scale.shape) == 1, "Expected per-axis scale to be 1D"
        else:
            assert len(self._scale.shape) == 0, "Expected per-tensor scale to be 0D"

    def dequantize_raw_tensor(
        self, t: torch.Tensor, to: torch.dtype, *, name: str
    ) -> torch.Tensor:
        return (
            PlanarQuantizedTensor(
                shape=t.shape,
                name=t.name,
                layout=TensorScaledLayout(
                    shape=t.shape,
                    d=self._reciprocal_scale,
                    qs=t,
                    m=self.offset,
                    dtype=to,
                ),
            )
            .unpack()
            .dequant()
        )

    def _quantize_raw_tensor(self, t: torch.Tensor, *, name: str) -> QuantizedTensor:
        """Performs a quantizing transformation on t, returning a QuantizeTensor."""
        shape = list(t.shape)
        axis = self._axis
        offset = self._offset
        if axis is None:
            # Per tensor.
            if offset is None:
                # Changed to t/reciprocal because narrow float types are garbage
                qs = saturate_cast(
                    t / self._reciprocal_scale,
                    dtype=self.dtype,
                    disable_saturate=self._disable_saturate,
                )
            else:
                qs = saturate_cast(
                    t / self._reciprocal_scale + offset,
                    dtype=self.dtype,
                    disable_saturate=self._disable_saturate,
                )
            return PlanarQuantizedTensor(
                shape=shape,
                name=name,
                layout=TensorScaledLayout(
                    shape=shape,
                    d=self._reciprocal_scale,
                    qs=qs,
                    m=self._offset,
                    dtype=t.dtype,  # Original dtype.
                ),
            )
        else:
            # Expand the scale/reciprocal to correspond to the broadcast axis.
            scale = self._scale
            reciprocal_scale = self._reciprocal_scale
            offset = self._offset
            assert axis >= 0 and axis < len(
                shape
            ), f"Per-axis scale {axis} out of bounds of shape {shape}"
            scale_shape = [1] * len(shape)
            scale_shape[axis] = scale.shape[0]
            broadcast_scale = scale.reshape(scale_shape)
            broadcast_reciprocal_scale = reciprocal_scale.reshape(scale_shape)
            if offset is None:
                broadcast_offset = None
                qs = saturate_cast(
                    t * broadcast_scale,
                    dtype=self.dtype,
                    disable_saturate=self._disable_saturate,
                )
            else:
                broadcast_offset = offset.reshape(scale_shape)
                qs = saturate_cast(
                    t * broadcast_scale + broadcast_offset,
                    dtype=self.dtype,
                    disable_saturate=self._disable_saturate,
                )
            return PlanarQuantizedTensor(
                shape=shape,
                name=name,
                layout=TensorScaledLayout(
                    shape=shape,
                    d=broadcast_reciprocal_scale,
                    qs=qs,
                    m=broadcast_offset,
                    dtype=t.dtype,  # Original dtype.
                ),
            )

    @property
    def axis(self) -> Optional[int]:
        """Returns the axis that is scaled or None for whole tensor."""
        return self._axis

    @property
    def offset(self) -> Optional[torch.Tensor]:
        return self._offset

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def reciprocal_scale(self) -> torch.Tensor:
        return self._reciprocal_scale

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def serialized_name(cls) -> str:
        return "StaticScaledQuantizer"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ):
        offset = None
        try:
            scale = raw_tensors["scale"]
            reciprocal_scale = raw_tensors["rscale"]
            if "offset" in raw_tensors:
                offset = raw_tensors["offset"]
        except KeyError as e:
            raise IOError("Missing component tensor 'scale'") from e
        try:
            dtype_name = extra_properties["dtype"]
        except KeyError as e:
            raise IOError("Missing property 'dtype' in extra_properties") from e
        axis = int(extra_properties["axis"]) if "axis" in extra_properties else None
        disable_saturate = bool(extra_properties.get("disable_saturate"))
        dtype = serialized_name_to_dtype(dtype_name)
        return cls(
            name=name,
            scale=scale,
            offset=offset,
            reciprocal_scale=reciprocal_scale,
            dtype=dtype,
            axis=axis,
            disable_saturate=disable_saturate,
        )

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        d = {
            f"{self.name}:scale": self._scale,
            f"{self.name}:rscale": self._reciprocal_scale,
        }
        if self._offset is not None:
            d[f"{self.name}:offset"] = self._offset
        return d

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        scale_name = f"{self.name}:scale"
        rscale_name = f"{self.name}:rscale"
        offset_name = f"{self.name}:offset"
        extra_properties = {"dtype": dtype_to_serialized_name(self._dtype)}
        if self._axis is not None:
            extra_properties["axis"] = self._axis
        if self._disable_saturate:
            extra_properties["disable_saturate"] = True
        raw_tensors = {
            "scale": scale_name,
            "rscale": rscale_name,
        }
        builder.add_tensor(scale_name, self._scale)
        builder.add_tensor(rscale_name, self._reciprocal_scale)
        if self._offset is not None:
            raw_tensors["offset"] = offset_name
            builder.add_tensor(offset_name, self._offset)

        return InferenceTensorMetadata(
            self.serialized_name(),
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        offset_name = f"{self.name}:offset"
        return StaticScaledQuantizer(
            name=self.name,
            dtype=self.dtype,
            axis=self.axis,
            disable_saturate=self._disable_saturate,
            scale=new_globals[f"{self.name}:scale"],
            reciprocal_scale=new_globals[f"{self.name}:rscale"],
            offset=new_globals.get(offset_name),
        )

    def __repr__(self):
        return (
            f"StaticScaledQuantizer({self.name}, {self.shape}, "
            f"scale=({self._scale.shape}, {self._scale.dtype}) along {self._axis}) "
            f"offset={self._offset} "
            f"-> dtype={self._dtype})"
        )


@register_inference_tensor
class DynamicScaledQuantizer(QuantizerTensor):
    """Quantizer that produced a `TensorScaledLayout` (per-tensor) based on
    computing the dynamic scale of the source tensor.

    This is done via a computation like:

    ```
    finfo = torch.finfo(output_dtype)
    amax = abs(max(x))
    scale = finfo.max / amax.clamp(eps)
    ```

    Note that this quantizer has only been used for testing and bringup, and
    it could use some more diligence done on the algorithm for determining
    scales in a dtype specific way.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        name: str = UnnamedTensorName,
    ):
        super().__init__(shape=(), name=name)
        self._dtype = dtype
        assert (
            dtype.is_floating_point or dtype.is_signed
        ), f"DynamicScaledQuantizer dtype must be fp or signed but got {dtype}"

    def _quantize_raw_tensor(self, t: torch.Tensor, *, name: str) -> QuantizedTensor:
        dtype = self._dtype
        amax = torch.max(torch.abs(t))
        if dtype.is_floating_point:
            finfo = torch.finfo(dtype)
            scale = finfo.max / amax.clamp(finfo.eps)
            reciprocal_scale = 1 / scale
            qs = saturate_cast(t * scale, self.dtype, round_int=True)
        else:
            eps = 1e-6
            iinfo = torch.iinfo(dtype)
            scale = iinfo.max / amax.clamp(eps)
            reciprocal_scale = 1.0 / scale
            qs = saturate_cast(t * scale, self.dtype, round_int=True)
        shape = list(t.shape)
        return PlanarQuantizedTensor(
            shape=shape,
            name=name,
            layout=TensorScaledLayout(
                shape=shape, d=reciprocal_scale, qs=qs, dtype=t.dtype  # Original dtype.
            ),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def serialized_name(cls) -> str:
        return "DynamicScaledQuantizer"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ):
        try:
            dtype_name = extra_properties["dtype"]
        except KeyError as e:
            raise IOError("Missing property 'dtype' in extra_properties") from e
        dtype = serialized_name_to_dtype(dtype_name)
        return cls(
            name=name,
            dtype=dtype,
        )

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {}

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        extra_properties = {"dtype": dtype_to_serialized_name(self._dtype)}
        raw_tensors = {}
        return InferenceTensorMetadata(
            self.serialized_name(),
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        return DynamicScaledQuantizer(
            name=self.name,
            dtype=self.dtype,
        )

    def __repr__(self):
        return f"DynamicScaledQuantizer({self.name}) " f"-> dtype={self._dtype})"


def pad_tensor_for_block_quantization(t: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pad tensor to make the last dimension evenly divisible by block_size.

    Args:
        t: Input tensor to pad
        block_size: Size of each block

    Returns:
        Padded tensor with shape such that t.shape[-1] % block_size == 0
    """
    if t.numel() == 0:
        raise ValueError("Cannot pad empty tensor")

    last_dim_size = t.shape[-1]
    pad_size = (block_size - (last_dim_size % block_size)) % block_size

    if pad_size > 0:
        return torch.nn.functional.pad(t, (0, pad_size))
    else:
        return t


def _fp4_block_quantize_tensor(
    t: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    use_fe8m0_scale: bool,
    name: str,
) -> PlanarQuantizedTensor:
    """Complete FP4 block quantization: blocking, scaling, quantization, and layout creation.

    Args:
        t: Input tensor of shape [..., N] to quantize (must have N % block_size == 0)
        scales: Per-block scales (either float or integer exponents, flat)
        block_size: Size of each block
        use_fe8m0_scale: Whether scales are FE8M0
        name: Name for the resulting tensor

    Returns:
        PlanarQuantizedTensor with BlockScaledFp4Layout
    """
    if t.numel() == 0:
        raise ValueError("Cannot quantize empty tensor")
    if t.shape[-1] % block_size != 0:
        raise ValueError(
            f"Tensor shape {t.shape[-1]} must be divisible by block_size {block_size}. "
            f"Use pad_tensor_for_block_quantization() to pad the tensor first."
        )

    # Reshape to [..., num_blocks, block_size] to group into blocks
    values_blocked = t.view(-1, block_size)

    # Prepare scales for broadcasting - add dimension for block_size
    if use_fe8m0_scale:
        scales_broadcast = e8m0_to_float32(scales).unsqueeze(-1)
    else:
        scales_broadcast = scales.unsqueeze(-1)

    # Scale the blocked values via broadcasting
    scaled_values = values_blocked / scales_broadcast

    # Convert to FP4 indices (preserves shape)
    quantized_indices = float32_to_fp4_e2m1(scaled_values)

    # Pack FP4 indices (works on last dimension)
    packed_fp4 = pack_fp4_e2m1_to_uint8(quantized_indices)

    # Create layout
    layout = BlockScaledFp4Layout(
        shape=list(t.shape),
        d=scales,
        qs=packed_fp4,
        block_size=block_size,
        use_fe8m0_scale=use_fe8m0_scale,
    )

    return PlanarQuantizedTensor(
        shape=list(t.shape),
        name=name,
        layout=layout,
    )


@register_inference_tensor
class StaticFp4BlockQuantizer(QuantizerTensor):
    """Quantizer that produces a `BlockScaledFp4Layout` with pre-computed static
    per-block scales, specifically designed for FP4 E2M1 quantization.
    """

    def __init__(
        self,
        *,
        scales: torch.Tensor,
        block_size: int = 32,
        use_fe8m0_scale: bool = True,
        dtype: torch.dtype = torch.float32,
        name: str = UnnamedTensorName,
    ):
        super().__init__(shape=scales.shape, name=name)
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}")
        if block_size % 2 != 0:
            raise ValueError(
                f"Block size must be even for FP4 packing, got {block_size}"
            )

        self._scales = scales
        self._block_size = block_size
        self._use_fe8m0_scale = use_fe8m0_scale
        self._dtype = dtype

    def _quantize_raw_tensor(self, t: torch.Tensor, *, name: str) -> QuantizedTensor:
        """Performs FP4 block quantization on tensor t using pre-computed scales."""

        return _fp4_block_quantize_tensor(
            t=t,
            scales=self.scales,
            block_size=self._block_size,
            use_fe8m0_scale=self._use_fe8m0_scale,
            name=name,
        )

    @property
    def scales(self) -> torch.Tensor:
        return self._scales

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def use_fe8m0_scale(self) -> bool:
        return self._use_fe8m0_scale

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def serialized_name(cls) -> str:
        return "StaticFp4BlockQuantizer"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ):
        try:
            scales = raw_tensors["scales"]
        except KeyError as e:
            raise IOError("Missing component tensor 'scales'") from e

        block_size = int(extra_properties.get("block_size", 32))
        use_fe8m0_scale = bool(extra_properties.get("use_fe8m0_scale", True))
        dtype_name = extra_properties.get("dtype", "float32")
        dtype = serialized_name_to_dtype(dtype_name)

        return cls(
            name=name,
            scales=scales,
            block_size=block_size,
            use_fe8m0_scale=use_fe8m0_scale,
            dtype=dtype,
        )

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {
            f"{self.name}:scales": self._scales,
        }

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        scales_name = f"{self.name}:scales"
        builder.add_tensor(scales_name, self._scales)

        extra_properties = {
            "block_size": self._block_size,
            "use_fe8m0_scale": self._use_fe8m0_scale,
            "dtype": dtype_to_serialized_name(self._dtype),
        }
        raw_tensors = {
            "scales": scales_name,
        }

        return InferenceTensorMetadata(
            self.serialized_name(),
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        return StaticFp4BlockQuantizer(
            name=self.name,
            scales=new_globals[f"{self.name}:scales"],
            block_size=self.block_size,
            use_fe8m0_scale=self.use_fe8m0_scale,
            dtype=self.dtype,
        )

    def __repr__(self):
        return (
            f"StaticFp4BlockQuantizer({self.name}, scales={self.scales.shape}, "
            f"block_size={self.block_size}, "
            f"use_fe8m0_scale={self.use_fe8m0_scale}, "
            f"dtype={self.dtype})"
        )


@register_inference_tensor
class DynamicFp4BlockQuantizer(QuantizerTensor):
    """Quantizer that produces a `BlockScaledFp4Layout` with dynamically computed
    per-block scales, specifically designed for FP4 E2M1 quantization.

    This quantizer:
    1. Divides the input tensor into blocks of `block_size` elements
    2. Computes a dynamic scale for each block based on the block's max absolute value
    3. Quantizes each block to FP4 E2M1 format using the block's scale
    4. Packs the FP4 values 2 per byte
    5. Returns a PlanarQuantizedTensor with BlockScaledFp4Layout
    """

    def __init__(
        self,
        *,
        block_size: int = 32,
        use_fe8m0_scale: bool = True,
        dtype: torch.dtype = torch.float32,
        name: str = UnnamedTensorName,
    ):
        super().__init__(shape=(), name=name)
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}")
        if block_size % 2 != 0:
            raise ValueError(
                f"Block size must be even for FP4 packing, got {block_size}"
            )
        self._block_size = block_size
        self._use_fe8m0_scale = use_fe8m0_scale
        self._dtype = dtype

    def _quantize_raw_tensor(self, t: torch.Tensor, *, name: str) -> QuantizedTensor:
        """Performs FP4 block quantization on tensor t."""
        t_padded = pad_tensor_for_block_quantization(t, self._block_size)

        # Compute scales per block
        values_blocked = t_padded.view(-1, self._block_size)
        block_max = torch.max(torch.abs(values_blocked), dim=1, keepdim=True)[0]
        scales, _ = compute_fp4_block_scales(
            block_max, self._use_fe8m0_scale, self._dtype
        )

        return _fp4_block_quantize_tensor(
            t=t_padded,
            scales=scales,
            block_size=self._block_size,
            use_fe8m0_scale=self._use_fe8m0_scale,
            name=name,
        )

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def use_fe8m0_scale(self) -> bool:
        return self._use_fe8m0_scale

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def serialized_name(cls) -> str:
        return "DynamicFp4BlockQuantizer"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ):
        block_size = int(extra_properties.get("block_size", 32))
        use_fe8m0_scale = bool(extra_properties.get("use_fe8m0_scale", True))
        return cls(
            name=name,
            block_size=block_size,
            use_fe8m0_scale=use_fe8m0_scale,
        )

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {}

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        extra_properties = {
            "block_size": self._block_size,
            "use_fe8m0_scale": self._use_fe8m0_scale,
        }
        raw_tensors = {}
        return InferenceTensorMetadata(
            self.serialized_name(),
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        return DynamicFp4BlockQuantizer(
            name=self.name,
            block_size=self.block_size,
            use_fe8m0_scale=self.use_fe8m0_scale,
        )

    def __repr__(self):
        return (
            f"DynamicFp4BlockQuantizer({self.name}, block_size={self.block_size}, "
            f"use_fe8m0_scale={self.use_fe8m0_scale})"
        )


def _norm_per_axis_param(
    axis: Optional[int], *params: torch.Tensor
) -> Tuple[Optional[int], List[torch.Tensor]]:
    """Per-axis params can be one of:

    * Scalar, indicating that they apply to all axes (axis = None).
    * 1D tensor of values and an axis != None.
    * Broadcasted tensor of values that has one non-unit dim corresponding to axis.

    If axis is None, then the case is inferred from the parameters.
    The normalized axis and parameters are returned.
    """
    # Infer based on shapes.
    if axis is None:
        required_rank = None
        for p in params:
            if p is None:
                continue
            rank = len(p.shape)
            if required_rank is None:
                if rank == 0:
                    axis = None
                    required_rank = 0
                else:
                    axis = _find_non_unit_axis(p)
                    required_rank = rank
            else:
                # Enforce.
                if rank != required_rank:
                    raise AssertionError(
                        f"Expected rank {required_rank} quant parameter but "
                        f"got {rank}: {p}"
                    )

    if axis is None:
        return axis, params
    else:
        return axis, [t.squeeze() if t is not None else None for t in params]


def _find_non_unit_axis(p: torch.Tensor) -> int:
    axis = None
    for i, dim in enumerate(p.shape):
        if dim == 1:
            continue
        else:
            if axis is not None:
                raise AssertionError(
                    f"Expected a single non-unit dim for parameter: {p.shape}"
                )
            axis = i
    return 0 if axis is None else axis
