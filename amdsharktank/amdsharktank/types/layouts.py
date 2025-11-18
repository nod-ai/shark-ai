# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Abstract layout structs describing various physical arrangements.

These are typically logical, planar layouts over some fundamental data types.
Concrete sub-classes implement any necessary physical to logical mapping.

While many of these layouts will have one or more vendor specific, custom
packed realizations as a QuantizedTensor subtype, each also has a generic
planar QuantizedTensor which carries its tensors unpacked.
"""

import math
import torch
import warnings

from abc import abstractmethod
from typing import Optional

from .tensors import (
    register_quantized_layout,
    MetaDataValueType,
    QuantizedLayout,
    dtype_to_serialized_name,
    serialized_name_to_dtype,
)

from .layout_utils import (
    pack_fp4_e2m1_to_uint8,
    promote_linear_i4_block_to_i8,
    promote_linear_i6_block_to_i8,
    unpack_uint8_to_fp4_e2m1,
)

from .ocp_floats import (
    fp4_e2m1_to_float32,
    convert_fp4_scales_to_float,
)

from amdsharktank.utils.misc import iterables_equal

__all__ = [
    "BlockScaledFp4Layout",
    "BlockScaledI4Layout",
    "BlockScaledLayout",
    "BlockScaledPackedLayout",
    "SuperBlockOffsetScaled_4_6_Layout",
    "TensorScaledLayout",
]


@register_quantized_layout
class TensorScaledLayout(QuantizedLayout):
    """Quantized layout which combines some scalar scale (`d`) tensor with a
    quantized sample (`qs`) tensor. An optional offset (`m`) tensor
    can be provided.

    The dequantization formula:

    ```
    dtype = d.dtype
    result = d.to(dtype) * (qs - m)
    ```

    If provided, `m` must be of the same dtype as `d`. `qs` must be cast
    compatible to `d.dtype`. Generally, `qs` will be a lower precision
    floating point format or an integer dtype.

    If d/m are scalar tensors, then this implements whole tensor quantization.
    Otherwise, they must be broadcast to the axis along which scaling is
    performed.

    If initialized with a dtype, the result of the conversion will be cast
    to this dtype (unless if otherwise specified in dequant()). If dtype is
    not specified in the constructor, it will default to the dtype of `d`.
    For low precision fp activation types, it can be necessary to have a higher
    precision `d`.
    """

    def __init__(
        self,
        *,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        m: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._shape = shape
        self._d = d
        self._qs = qs
        self._m = m
        self._dtype = dtype if dtype is not None else d.dtype

    @classmethod
    def serialized_name(cls) -> str:
        return "TensorScaledLayout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        m = planes.get("m")
        dtype_str = metadata.get("dtype")
        if dtype_str is not None:
            dtype = serialized_name_to_dtype(dtype_str)
        else:
            # Backwards compat with old serialized. Emulate original behavior
            # before mixed precision.
            dtype = None
        return cls(shape=shape, d=planes["d"], qs=planes["qs"], m=m, dtype=dtype)

    @property
    def metadata(self) -> Optional[dict[str, MetaDataValueType]]:
        """Additional metadata needed to reconstruct a layout."""
        return {"dtype": dtype_to_serialized_name(self._dtype)}

    @property
    def planes(self) -> dict[str, torch.Tensor]:
        p = {
            "d": self._d,
            "qs": self._qs,
        }
        if self._m is not None:
            p["m"] = self._m
        return p

    @property
    def shape(self) -> list[int]:
        """The flattened shape of the logical (unblocked) result."""
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def d(self) -> torch.Tensor:
        """Per tensor scale."""
        return self._d

    @property
    def m(self) -> Optional[torch.Tensor]:
        """Per tensor offset."""
        return self._m

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values."""
        return self._qs

    def view(self, *args, **kwargs):
        qs = self.qs.view(*args, **kwargs)
        return TensorScaledLayout(
            shape=qs.shape, d=self.d, qs=qs, m=self.m, dtype=self.dtype
        )

    def flatten(self, *args, **kwargs):
        qs = self.qs.flatten(*args, **kwargs)
        return TensorScaledLayout(
            shape=qs.shape, d=self.d, qs=qs, m=self.m, dtype=self.dtype
        )

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        m = self.m
        qs = self.qs
        if dtype is None:
            dtype = self.dtype
        rescale_dtype = d.dtype
        qs = qs.to(dtype=rescale_dtype)
        if m is not None:
            m = m.to(rescale_dtype)
            result = (qs - m) * d
        else:
            result = qs * d
        if result.dtype != dtype:
            result = result.to(dtype=dtype)
        return result

    def __repr__(self):
        r = (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"qs({list(self.qs.shape)}, dtype={self.qs.dtype}))"
        )
        if self.m is not None:
            r += f", m({list(self.m.shape)}, dtype={self.m.dtype})"
        r += f" -> {self.dtype}"
        return r


@register_quantized_layout
class BlockScaledLayout(QuantizedLayout):
    """Block-quantized representation which consists of a scale (`d`)
    and offset (`m`) per block in a higher precision type. The offset, if
    present, is pre-scaled.

    The dequantization formula:

    ```
    result = d.to(dtype) * qs.to(dtype) + m.to(dtype)
    ```

    The inner-most dims will retain block structure. For example, if the
    block size is 32 and the original shape was NxK, then the component
    shapes would be:

    * `d`: `[N, K // 32, 1]`
    * `m`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`

    Note that the offset (`m`) is optional.
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        *,
        m: Optional[torch.Tensor] = None,
    ):
        self._shape = shape
        self._d = d
        self._qs = qs
        self._m = m

    @classmethod
    def serialized_name(cls) -> str:
        return "BlockScaledLayout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        m = planes.get("m")
        return cls(shape, planes["d"], planes["qs"], m=m)

    @property
    def planes(self) -> dict[str, torch.Tensor]:
        p = {
            "d": self._d,
            "qs": self._qs,
        }
        if hasattr(self, "_m") and self._m is not None:
            p["m"] = self._m
        return p

    @property
    def shape(self) -> list[int]:
        """The flattened shape of the logical (unblocked) result."""
        return self._shape

    @property
    def d(self) -> torch.Tensor:
        """Per block scales."""
        return self._d

    @d.setter
    def d(self, value: torch.Tensor):
        self._d = value

    @property
    def m(self) -> torch.Tensor:
        """Per block offsets."""
        return self._m

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values."""
        return self._qs

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        m = self.m
        qs = self.qs
        if dtype:
            d = d.to(dtype)
            if m is not None:
                m = m.to(dtype)
        else:
            dtype = d.dtype
            assert m is None or m.dtype == d.dtype
        scaled = d * qs.to(dtype)
        shifted = scaled if m is None else scaled + m
        return shifted

    def __repr__(self):
        r = (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"qs({list(self.qs.shape)}, dtype={self.qs.dtype}))"
        )
        if self.m is not None:
            r += f", m({list(self.m.shape)}, dtype={self.m.dtype})"
        return r


class BlockScaledPackedLayout(BlockScaledLayout):
    """Base class for block-scaled layouts with packed quantized values.

    This abstract base class extends BlockScaledLayout for formats that use packed sub-byte quantized values

    Subclasses must implement qs() describing how to unpack the raw data.
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs_packed: torch.Tensor,
        *,
        m: Optional[torch.Tensor] = None,
    ):
        super().__init__(shape, d, qs_packed, m=m)

    @property
    def qs_bit_packed(self) -> torch.Tensor:
        """Gets the qs as a bit-packed tensor"""
        return self._qs

    @property
    def qs(self) -> torch.Tensor:
        """Logical values (unpacked)."""
        return self.unpack_qs(self._qs)

    @abstractmethod
    def pack_qs(self, qs: torch.Tensor) -> torch.Tensor:
        """Pack the logical values into the underlying bit-packed tensor."""
        ...

    def unpack_qs(self, qs: torch.Tensor) -> torch.Tensor:
        """Unpack the underlying bit-packed tensor into logical values."""
        from amdsharktank import ops

        return ops.unpack_qs(qs, self)


@register_quantized_layout
class BlockScaledI4Layout(BlockScaledPackedLayout):
    """A BlockScaledLayout where the `qs` are internally packed 2 values per byte.

    Per convention, the `qs` property returns a tensor as either uint8 or
    int8 (depending on `signed=`) that can be used directly for arithmetic.
    The underlying bit-packed tensor can be accessed via `qs_bit_packed` and
    it is laid out in little endian bit order, linearly across the block
    dimension. There are an arbitrary ways to organize such things, and
    if more specificity is needed, a dedicated layout class should be used. In
    general, for these "generic" layouts, we choose defaults that mate well
    with how the compiler infra and prevailing targets are built and trust that
    optimizations that care will choose a specific packing.
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        *,
        m: Optional[torch.Tensor] = None,
        signed: bool = False,
    ):
        super().__init__(shape, d, qs, m=m)
        self.signed = signed

    @classmethod
    def serialized_name(cls) -> str:
        return "BlockScaledI4Layout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        m = planes.get("m")
        return cls(shape, planes["d"], planes["qs"], m=m, signed=metadata["signed"])

    @property
    def metadata(self) -> dict[str, MetaDataValueType]:
        return {"signed": self.signed}

    def pack_qs(self, qs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Need inverse of promote_linear_i4_block_to_i8")

    def unpack_qs(self, qs: torch.Tensor) -> torch.Tensor:
        """Unpack the underlying bit-packed tensor into logical values."""
        return promote_linear_i4_block_to_i8(qs, signed=self.signed)


@register_quantized_layout
class SuperBlockOffsetScaled_4_6_Layout(QuantizedLayout):
    """Effectively a planarized version of the ggml Q4_K layout."""

    def __init__(
        self,
        shape: list[int],
        *,
        d: torch.Tensor,
        dmin: torch.Tensor,
        sb_scales_high: torch.Tensor,
        sb_scales_low: torch.Tensor,
        sb_mins_high: torch.Tensor,
        sb_mins_low: torch.Tensor,
        qs: torch.Tensor,
    ):
        self._shape = shape
        self._d = d
        self._dmin = dmin
        self._sb_scales_high = sb_scales_high
        self._sb_scales_low = sb_scales_low
        self._sb_mins_high = sb_mins_high
        self._sb_mins_low = sb_mins_low
        self._qs = qs

    @classmethod
    def serialized_name(cls) -> str:
        return "SuperBlockOffsetScaled_4_6_Layout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        return cls(
            shape,
            d=planes["d"],
            dmin=planes["dmin"],
            sb_scales_high=planes["sb_scales_high"],
            sb_scales_low=planes["sb_scales_low"],
            sb_mins_high=planes["sb_mins_high"],
            sb_mins_low=planes["sb_mins_low"],
            qs=planes["qs"],
        )

    @property
    def planes(self) -> dict[str, torch.Tensor]:
        return {
            "d": self._d,
            "dmin": self._dmin,
            "sb_scales_high": self._sb_scales_high,
            "sb_scales_low": self._sb_scales_low,
            "sb_mins_high": self._sb_mins_high,
            "sb_mins_low": self._sb_mins_low,
            "qs": self._qs,
        }

    @property
    def shape(self) -> list[int]:
        """The flattened shape of the logical (unblocked) result.

        Shape: [N, SUPER_COUNT * SUB_COUNT * BLOCK_SIZE]
        """
        return self._shape

    @property
    def d(self) -> torch.Tensor:
        """Super-block scales.

        Shape: [N, SUPER_COUNT, 1]
        """
        return self._d

    @property
    def dmin(self) -> torch.Tensor:
        """Super-block mins.

        Shape: [N, SUPER_COUNT, 1]
        """
        return self._dmin

    @property
    def sb_scales(self) -> torch.Tensor:
        """Returns sub-block scales combined and cast to a uint8 tensor.

        Shape: [N, SUPER_COUNT, SUB_COUNT]
        """
        return promote_linear_i6_block_to_i8(self._sb_scales_high, self._sb_scales_low)

    @property
    def sb_mins(self) -> torch.Tensor:
        """Returns sub-block mins combined and cast to a uint8 tensor.

        Shape: [N, SUPER_COUNT, SUB_COUNT]
        """
        return promote_linear_i6_block_to_i8(self._sb_mins_high, self._sb_mins_low)

    @property
    def sb_scales_bit_packed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Bit packed sub-block scales.

        Shape:
            high = [N, SUPER_COUNT, SUB_COUNT // 4]
            low  = [N, SUPER_COUNT, SUB_COUNT // 2]

        The 'high' tensor contains upper 2 bits of each. The 'low' tensor
        contains the lower nibble.
        """
        return self._sb_scales_high, self._sb_scales_low

    @property
    def sb_mins_bit_packed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Bit packed sub-block mins.

        Shape:
            high = [N, SUPER_COUNT, SUB_COUNT // 4]
            low  = [N, SUPER_COUNT, SUB_COUNT // 2]

        The 'high' tensor contains upper 2 bits of each. The 'low' tensor
        contains the lower nibble.
        """
        return self._sb_mins_high, self._sb_mins_low

    @property
    def qs_bit_packed(self) -> torch.Tensor:
        """Gets the qs as a bit-packed i4 tensor (as uint8).

        Shape: [N, SUPER_COUNT, SUB_COUNT, BLOCK_SIZE // 2]
        """
        return self._qs

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values.

        Shape: [N, SUPER_COUNT, SUB_COUNT, BLOCK_SIZE]
        """
        # `qs` is defined as something that we can do integer arithmetic on
        # for cases where we only have non-packed kernels available. Therefore,
        # we promote it to i8. The `qs_packed` is available for the sub-byte
        # bit pattern.
        return promote_linear_i4_block_to_i8(self._qs, signed=False)

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        dmin = self.dmin
        qs = self.qs
        sb_scales = self.sb_scales
        sb_mins = self.sb_mins

        d_scaled = (d * sb_scales).unsqueeze(-1)
        dmin_scaled = (dmin * sb_mins).unsqueeze(-1)
        return d_scaled * qs - dmin_scaled

    def __repr__(self):
        r = (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"dmin({list(self.dmin.shape)}, dtype={self.dmin.dtype}), "
            f"sb_scales_high({list(self._sb_scales_high.shape)}, dtype={self._sb_scales_high.dtype}), "
            f"sb_scales_low({list(self._sb_scales_low.shape)}, dtype={self._sb_scales_low.dtype}), "
            f"sb_mins_high({list(self._sb_mins_high.shape)}, dtype={self._sb_mins_high.dtype}), "
            f"sb_mins_low({list(self._sb_mins_low.shape)}, dtype={self._sb_mins_low.dtype}), "
            f"qs({list(self._qs.shape)}, dtype={self._qs.dtype}))"
        )
        return r


@register_quantized_layout
class BlockScaledFp4Layout(BlockScaledPackedLayout):
    """Block-quantized FP4 E2M1 representation

    This layout is specifically designed for FP4 E2M1 block quantization where:
    - FP4 indices are packed 2 per byte in the `qs` tensor
    - Scales can be either FE8M0 (stored as integer exponents) or regular floats
    - Each block has its own scale for better accuracy


    The inner-most dims will retain block structure. For example, if the
    block size is 32 and the original shape was NxK, then the component
    shapes would be:

    * `d`: `[N, K // 32, 1]` (per-block scales)
    * `qs`: `[N, K // 32, 16]` (packed FP4 indices, 32 values packed into 16 bytes)
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        *,
        block_size: int = 32,
        use_fe8m0_scale: bool = True,
    ):
        if len(qs.shape) == len(d.shape) + 1:
            # Legacy scale format with no trailing singleton dimension to match qs.
            # This is here to avoid breaking existing IRPA files.
            warnings.warn(
                (
                    "Constructing BlockScaledFp4Layout with scales tensor of shape "
                    f"{d.shape} without a trailing singleton dimension is deprecated. "
                    "Maybe you are using an old model file (IRPA)."
                ),
                DeprecationWarning,
            )
            d = d.unsqueeze(-1)
        assert iterables_equal(qs.shape[:-1], d.shape[:-1])
        assert math.prod(shape) == math.prod(qs.shape) * 2
        assert qs.shape[-1] * 2 == block_size
        super().__init__(shape=shape, d=d, qs_packed=qs)
        self._block_size = block_size
        self._use_fe8m0_scale = use_fe8m0_scale

    @classmethod
    def serialized_name(cls) -> str:
        return "BlockScaledFp4Layout"

    @classmethod
    def create(
        cls,
        shape: list[int],
        metadata: dict[str, MetaDataValueType],
        planes: dict[str, torch.Tensor],
    ):
        block_size = metadata.get("block_size", 32)
        use_fe8m0_scale = metadata.get("use_fe8m0_scale", True)
        res = BlockScaledFp4Layout(
            shape,
            planes["d"],
            planes["qs"],
            block_size=block_size,
            use_fe8m0_scale=use_fe8m0_scale,
        )

        if planes["d"] is not res.d:
            from iree.turbine.aot import ExternalTensorTrait

            external_tensor_trait = ExternalTensorTrait.get(planes["d"])
            if external_tensor_trait is not None:
                warnings.warn(
                    (
                        "Constructing BlockScaledFp4Layout requires retargeting the "
                        "ExternalTensorTrait of the d (scale) tensor. Maybe you are "
                        "using an old model file (IRPA)."
                    ),
                    DeprecationWarning,
                )
                ExternalTensorTrait(
                    external_tensor_trait.external_scope,
                    external_tensor_trait.external_name,
                ).set(res.d)

        return res

    @property
    def metadata(self) -> dict[str, MetaDataValueType]:
        return {
            "block_size": self._block_size,
            "use_fe8m0_scale": self._use_fe8m0_scale,
        }

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def use_fe8m0_scale(self) -> bool:
        """Whether scales are FE8M0 (integer exponents)."""
        return self._use_fe8m0_scale

    def pack_qs(self, qs: torch.Tensor) -> torch.Tensor:
        return pack_fp4_e2m1_to_uint8(qs)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.float32

        fp4_indices = unpack_uint8_to_fp4_e2m1(self._qs)
        fp4_as_float = fp4_e2m1_to_float32(fp4_indices)

        # Scale each block
        scales_float = convert_fp4_scales_to_float(self.d, self.use_fe8m0_scale)
        dequantized_blocked = (
            fp4_as_float * scales_float
        )  # Shape: [num_blocks, block_size]

        if dequantized_blocked.dtype != dtype:
            dequantized_blocked = dequantized_blocked.to(dtype=dtype)

        return dequantized_blocked

    def __repr__(self):
        return (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"qs({list(self._qs.shape)}, dtype={self._qs.dtype}), "
            f"block_size={self.block_size}, use_fe8m0_scale={self.use_fe8m0_scale})"
        )
