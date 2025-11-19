# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    TypeVar,
    Generic,
    Type,
    Iterable,
    List,
    Tuple,
    overload,
)
from copy import deepcopy
from collections.abc import Collection, Mapping, Sequence
from numbers import Integral, Number

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
import torch._subclasses.functional_tensor
from torch.utils._pytree import register_pytree_node, SequenceKey
import torch.utils._pytree
from amdsharktank.utils.math import ceildiv
from amdsharktank.utils import iterables_equal, tree as tree_utils
from amdsharktank.utils.io import ShardedArchiveBuilder
from iree.turbine.aot import (
    DeviceTensorTrait,
    ExternalTensorTrait,
)


__all__ = [
    "AnyTensor",
    "AnyTensorTree",
    "DefaultPrimitiveTensor",
    "dtype_to_serialized_name",
    "dtype_to_serialized_short_name",
    "flatten_tensor_tree",
    "InferenceTensor",
    "is_any_tensor",
    "MetaDataValueType",
    "PlanarQuantizedTensor",
    "PrimitiveTensor",
    "QuantizedLayout",
    "QuantizedTensor",
    "register_quantized_layout",
    "ReplicatedTensor",
    "serialized_name_to_dtype",
    "serialized_short_name_to_dtype",
    "ShardedTensor",
    "SplitPrimitiveTensor",
    "torch_tree_flatten",
    "unbox_tensor",
    "UnnamedTensorName",
    "UnreducedTensor",
]

if (
    "amdsharkTANK_OVERRIDE_TORCH_TENSOR_REPR" in os.environ
    and os.environ["amdsharkTANK_OVERRIDE_TORCH_TENSOR_REPR"] != "0"
):

    def _tensor_debugger_friendly_repr(self: torch.Tensor):
        """Override for the torch.Tensor.__repr__ so it does not take forever when the
        debugger wants to query many/large tensors."""
        return f"Tensor({list(self.shape)}, {self.dtype})"

    Tensor.__repr__ = _tensor_debugger_friendly_repr

# JSON encodable value types.
MetaDataValueType = Union[int, bool, float, str]
UnnamedTensorName = "<unnamed>"


class QuantizedLayout(ABC):
    @abstractmethod
    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        from amdsharktank import ops

        return ops.dequantize(self, dtype=dtype)

    @classmethod
    @abstractmethod
    def serialized_name(cls) -> str:
        """Returns the globally unique serialization name for this layout."""
        ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        shape: list[int],
        metadata: Optional[dict[str, MetaDataValueType]],
        planes: dict[str, torch.Tensor],
    ) -> "QuantizedLayout":
        ...

    @property
    @abstractmethod
    def planes(self) -> dict[str, torch.Tensor]:
        """Returns the planes of this layout as concrete, named tensors.

        When transforming, the name will form a local suffix (i.e. ":name")
        for stored values by combining the global name with the ":" separator.

        NOTE: Any bitpacked values will NOT be unpacked by this method.
        """
        ...

    @property
    def metadata(self) -> Optional[dict[str, MetaDataValueType]]:
        """Additional metadata needed to reconstruct a layout."""
        return None

    @property
    @abstractmethod
    def shape(self) -> list[int]:
        """The flattened shape of the logical result."""
        ...


QuantizedLayoutT = TypeVar("QuantizedLayoutT", bound=QuantizedLayout)


REGISTERED_LAYOUT_CLASSES: dict[str, Type[QuantizedLayout]] = {}


def register_quantized_layout(ty: Type[QuantizedLayoutT]) -> Type[QuantizedLayoutT]:
    """Class decorator which registers a serializable layout class."""
    name = ty.serialized_name()
    existing = REGISTERED_LAYOUT_CLASSES.get(name)
    assert (
        existing is None
    ), f"Duplicate QuantizedLayoutRegistration '{name}' ({ty} vs {existing})"
    REGISTERED_LAYOUT_CLASSES[name] = ty
    return ty


@dataclass
class InferenceTensorMetadata:
    # Registered name of an InferenceTensor subclass.
    type_name: str
    # Mapping of constituent local names to parameter archive global names
    # of individual tensors that make up this InferenceTensor.
    raw_tensors: dict[str, str]
    # Additional properties needed to restore the instance. Must be JSON
    # legal types. Will be added to the root JSON dictionary.
    extra_properties: Optional[dict[str, Any]] = None

    def create_instance(self) -> "InferenceTensor":
        try:
            clazz = REGISTERED_INFERENCE_TENSOR_CLASSES[self.type_name]
        except KeyError as e:
            raise IOError(
                f"Unable to create instance of unregistered type {self.type_name}"
            ) from e
        assert issubclass(clazz, InferenceTensor)

    def to_json(self) -> dict:
        d = {
            "type_name": self.type_name,
            "raw_tensors": self.raw_tensors,
        }
        if self.extra_properties is not None:
            d.update(self.extra_properties)
        return d

    def from_json(obj: dict) -> "InferenceTensorMetadata":
        extra_properties = dict(obj)
        try:
            type_name = extra_properties["type_name"]
            assert isinstance(type_name, str)
            del extra_properties["type_name"]
            raw_tensors = extra_properties["raw_tensors"]
            assert isinstance(raw_tensors, dict)
            del extra_properties["raw_tensors"]
        except Exception as e:
            raise IOError(f"Error decoding InferenceTensorMetadata object") from e

        # Validate.
        for k, v in raw_tensors.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise IOError(
                    f"Bad format for InferenceTensorMetadata.raw_tensors ({type(k)}, {type(v)})"
                )

        return InferenceTensorMetadata(
            type_name=type_name,
            raw_tensors=raw_tensors,
            extra_properties=extra_properties,
        )


class InferenceTensor(ABC):
    """Provides access to a tensor in the model used for inference.

    InferenceTensors have a richer structure than "normal" training tensors
    since they often involve a degree of layout on top of the raw data tensor.
    """

    def __init__(self, *, shape: list[int], name: str = UnnamedTensorName):
        self._name = name
        self.shape = shape

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        raise NotImplementedError(
            f"InferenceTensor {cls} cannot be deserialized "
            f"because it does not implement create()"
        )

    @classmethod
    def serialized_name(cls) -> str:
        """Returns the globally unique serialization name for this type."""
        raise NotImplementedError(
            f"InferenceTensor {cls} cannot be directly "
            f"serialized (does not implement serialized_name())"
        )

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    @abstractmethod
    def subtensors(self) -> dict[str, torch.Tensor]:
        """Returns a mapping of global name to root tensor.

        The primary accessors on an InferenceTensor access the root tensors in
        the global set, all of which in a root Theta must have unique names.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> InferenceTensorMetadata:
        """Gets the meta data for this inference tensor"""
        ...

    @abstractmethod
    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        ...

    def is_deep_equal(self, other: Any, *, compare_name: bool = True) -> bool:
        """Deep equality including metadata and exact equality of tensor elements.
        It is a representational equality."""
        raise NotImplementedError()

    def transform_subtensors(
        self,
        *transforms: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        copy_external_tensor_trait: bool = True,
    ) -> "InferenceTensor":
        """Appplies transformation functions to the InferenceTensors backing
        tensors.

        Args:
            transforms: A sequence of transformation functions.
                        Each transformation must produce a new dict of a form that the subclass can handle.
                        Practically, this means that placement and layout related changes are always allowed,
                        while more invasive changes (like dtype) are more case by case.
            copy_external_tensor_trait: If True, will copy the ExternalTensorTrait from the previous subtensors to the new ones, if it exists.

        Returns:
            A new InferenceTensor, mutated.
        """
        prev_subtensors = self.subtensors
        for transform in transforms:
            next_subtensors = transform(prev_subtensors)
            if copy_external_tensor_trait:
                # Copy any metadata from prior to next.
                for k, prev_t in prev_subtensors.items():
                    new_t = next_subtensors.get(k)
                    if new_t is None:
                        continue
                    if new_t is not prev_t:
                        ext_trait = ExternalTensorTrait.get(prev_t)
                        if ext_trait is not None:
                            ext_trait.set(new_t)
            prev_subtensors = next_subtensors
        return self._clone_with_subtensors(prev_subtensors)

    @overload
    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> "InferenceTensor":
        ...

    @overload
    def to(
        self,
        other: "AnyTensor",
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> "InferenceTensor":
        ...

    @overload
    def to(
        self,
        dtype: torch.dtype,
        non_blocking: bool = False,
        copy: bool = False,
        *,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> "InferenceTensor":
        ...

    def to(self, *args, **kwargs) -> "InferenceTensor":
        arg0 = args[0] if len(args) > 0 else None
        if arg0 is None and len(kwargs) == 0:
            return self
        device_overload = ("device" in kwargs) or isinstance(arg0, (str, torch.device))
        other_overload = ("other" in kwargs) or isinstance(arg0, AnyTensor)
        memory_overload = (
            ("memory_format" in kwargs)
            or ("dtype" in kwargs)
            or isinstance(arg0, torch.dtype)
        )

        if device_overload:
            # Do we always want to clone with globals?
            # This makes our type inconsistent with torch tensors.
            # If we use this to transform a theta we want to change the theta.
            # If we want to use this in a computation we don't want to change the theta.
            return self.transform_subtensors(
                lambda d: {k: t.to(*args, **kwargs) for k, t in d.items()}
            )
        elif other_overload:
            args = tuple([arg0.device, arg0.dtype] + list(args[1:]))
            return self.to(*args, **kwargs)
        elif memory_overload:
            from amdsharktank.ops import to

            return to(self, *args, **kwargs)

        raise ValueError(
            f"Could not idenify which overload to use given args, and kwargs: {args}{kwargs}"
        )

    def _clone_with_subtensors(
        self, new_subtensors: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        """
        Creates a clone of this InferenceTensor but with new subtensors.
        Other properties (like name, shape, etc.) are copied over from this InferenceTensor.

        Args:
            new_subtensors: A mapping of names to new underlying torch.Tensors that make up this InferenceTensor.
                            The names must match the original InferenceTensor's subtensors.
                            If the InferenceTensor is a global tensor from a Theta, then the names must be their unique global names.
        Returns:
            A new InferenceTensor with the same type as this one, but with the new subtensors.
        """
        raise NotImplementedError(
            f"InferenceTensor {type(self)} does not implement _clone_with_subtensors"
        )

    @property
    def T(self) -> "InferenceTensor":
        from amdsharktank.ops import permute

        # Reverse the dimension range.
        rank = len(self.shape)
        assert rank == 2, "T will be deprecated in torch for non-2D tensors"
        dims = [rank - 1 - i for i in range(rank)]

        return permute(self, dims=dims)

    @property
    def mT(self) -> "AnyTensor":
        from amdsharktank.ops import transpose

        return transpose(self, -2, -1)

    def abs(self) -> "AnyTensor":
        from amdsharktank.ops import abs

        return abs(self)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["AnyTensor", ...]:
        from amdsharktank.ops import chunk

        return chunk(self, chunks, dim)

    def contiguous(self) -> "InferenceTensor":
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        """Equivalent to torch.Tensor.device."""
        raise NotImplementedError()

    def dim(self) -> int:
        raise NotImplementedError()

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError()

    def expand(self, *args: Union[List[List[int]], List[int]]) -> "AnyTensor":
        from amdsharktank.ops import expand

        if all(isinstance(a, int) for a in args):
            shape = args
        else:
            assert len(args) == 1
            shape = args[0]
        return expand(self, shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "AnyTensor":
        from amdsharktank.ops import flatten

        return flatten(self, start_dim, end_dim)

    def log(self) -> "AnyTensor":
        from amdsharktank.ops import log

        return log(self)

    def index_copy_(
        self, dim: int, index: "AnyTensor", tensor: "AnyTensor"
    ) -> "InferenceTensor":
        from amdsharktank.ops import index_copy_

        return index_copy_(self, dim, index, tensor)

    def index_put_(
        self, indices: Tuple["AnyTensor"], values: "AnyTensor"
    ) -> "InferenceTensor":
        from amdsharktank.ops import index_put_

        return index_put_(self, indices, values)

    def index_select(
        self,
        dim: int,
        index: "AnyTensor",
    ) -> "InferenceTensor":
        from amdsharktank.ops import index_select

        return index_select(self, dim, index)

    def masked_fill(self, mask: "AnyTensor", value: Number) -> "InferenceTensor":
        from amdsharktank.ops import masked_fill

        return masked_fill(self, mask, value)

    def mean(
        self,
        dim: Union[int, List[int]],
        keepdim: bool = False,
        *,
        dtype: torch.dtype = None,
    ) -> "AnyTensor":
        from amdsharktank.ops import mean

        return mean(self, dim, keepdim, dtype=None)

    @property
    def ndim(self) -> int:
        return self.dim()

    def permute(self, dims: List[int]) -> "AnyTensor":
        from amdsharktank.ops import permute

        return permute(self, dims)

    def pow(self, exponent: Union["AnyTensor", Number]) -> "AnyTensor":
        from amdsharktank.ops import elementwise

        return elementwise(torch.pow, self, exponent)

    def repeat(self, *sizes: List[int]) -> "AnyTensor":
        from amdsharktank.ops import repeat

        return repeat(self, *sizes)

    def reshape(self, *args: Union[List[List[int]], List[int]]) -> "AnyTensor":
        from amdsharktank.ops import reshape

        if all(isinstance(a, (int, torch.SymInt)) for a in args):
            shape = args
        else:
            assert len(args) == 1
            shape = args[0]
        return reshape(self, shape)

    def scatter_(
        self,
        dim: int,
        index: "AnyTensor",
        src: Union["AnyTensor", Number],
        *,
        reduce=None,
    ) -> "AnyTensor":
        from amdsharktank.ops import scatter_

        return scatter_(self, dim, index, src, reduce=reduce)

    def scatter_add(
        self, dim: int, index: "AnyTensor", src: "AnyTensor"
    ) -> "AnyTensor":
        from amdsharktank.ops import scatter_add

        return scatter_add(self, dim, index, src)

    def sigmoid(self) -> "AnyTensor":
        from amdsharktank.ops import sigmoid

        return sigmoid(self)

    def sin(self) -> "AnyTensor":
        from amdsharktank.ops import sin

        return sin(self)

    def cos(self) -> "AnyTensor":
        from amdsharktank.ops import cos

        return cos(self)

    def size(self, dim: Optional[int] = None) -> tuple[int]:
        if dim is None:
            return tuple(self.shape)
        return self.shape[dim]

    def softmax(
        self, dim: Optional[int] = None, dtype: Optional[torch.dtype] = None
    ) -> "AnyTensor":
        from amdsharktank.ops import softmax

        return softmax(self, dim, dtype=dtype)

    def split(
        self, split_size_or_sections: int | list[int], dim: int = 0
    ) -> tuple["AnyTensor", ...]:
        from amdsharktank.ops import split

        return split(self, split_size_or_sections, dim)

    def squeeze(self, dim: Optional[int] = None) -> "AnyTensor":
        from amdsharktank.ops import squeeze

        return squeeze(self, dim)

    def squeeze(self, dim: Optional[int] = None) -> "AnyTensor":
        from amdsharktank.ops import squeeze

        return squeeze(self, dim)

    def sum(
        self,
        dim: Union[int, List[int], None] = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype = None,
    ) -> "AnyTensor":
        from amdsharktank.ops import sum

        return sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def topk(
        self, k: int, dim: int, largest: bool = True, sorted: bool = True
    ) -> Tuple["AnyTensor"]:
        from amdsharktank.ops import topk

        return topk(self, k, dim, largest, sorted)

    def transpose(self, dim0: int, dim1: int) -> "AnyTensor":
        from amdsharktank.ops import transpose

        return transpose(self, dim0, dim1)

    def unflatten(self, dim: int, sizes: Tuple[int]) -> "AnyTensor":
        from amdsharktank.ops import unflatten

        return unflatten(self, dim, sizes)

    def unsqueeze(self, dim: int) -> "AnyTensor":
        from amdsharktank.ops import unsqueeze

        return unsqueeze(self, dim)

    @overload
    def view(self, dtype: torch.dtype) -> "AnyTensor":
        ...

    @overload
    def view(self, *args: Union[List[List[int]], List[int]]) -> "AnyTensor":
        ...

    def view(
        self,
        *args: Union[List[List[int]], List[int], torch.dtype],
        dtype: torch.dtype | None = None,
    ) -> "AnyTensor":
        from amdsharktank.ops import view

        shape = None

        if len(args) > 0 and all(
            isinstance(a, int) or isinstance(a, torch.SymInt) for a in args
        ):
            shape = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                assert dtype is None
                dtype = args[0]
            else:
                assert isinstance(args[0], Sequence)
                shape = args[0]
        else:
            assert dtype is not None

        return view(self, shape=shape, dtype=dtype)

    def __gt__(self, lhs: Union["AnyTensor", Number]) -> "AnyTensor":
        from amdsharktank.ops import elementwise
        from operator import gt

        return elementwise(gt, self, lhs)

    def __add__(self, rhs):
        from amdsharktank.ops import elementwise

        return elementwise(torch.add, self, rhs)

    def __radd__(self, lhs):
        # Assumes commutative addition due to torch elementwise ops not handling
        # numbers on the lhs.
        return self.__add__(lhs)

    def __sub__(self, rhs):
        from amdsharktank.ops import elementwise

        return elementwise(torch.sub, self, rhs)

    def __rsub__(self, lhs):
        # Assumes commutative addition due to torch elementwise ops not handling
        # numbers on the lhs.
        return self.__sub__(lhs)

    def __mod__(self, rhs):
        from amdsharktank.ops import elementwise

        return elementwise(torch.remainder, self, rhs)

    def __mul__(self, rhs):
        from amdsharktank.ops import elementwise

        return elementwise(torch.mul, self, rhs)

    def __matmul__(self, rhs):
        from amdsharktank.ops import matmul

        return matmul(self, rhs)

    def __rmatmul__(self, lhs):
        from amdsharktank.ops import matmul

        return matmul(lhs, self)

    def __rmul__(self, lhs):
        # Assumes commutative multiplication due to torch elementwise ops not handling
        # numbers on the lhs.
        return self.__mul__(lhs)

    def __truediv__(self, rhs):
        from amdsharktank.ops import elementwise

        return elementwise(torch.true_divide, self, rhs)

    def __floordiv__(self, rhs):
        from amdsharktank.ops import elementwise

        return elementwise(torch.floor_divide, self, rhs)

    def __getitem__(self, key):
        from amdsharktank.ops import extract_slice

        return extract_slice(self, key)

    def _is_deep_equal(self, other: Any, compare_name: bool = True) -> bool:
        if self.shape != other.shape:
            return False
        if compare_name and self.name != other.name:
            return False
        return True


REGISTERED_INFERENCE_TENSOR_CLASSES: dict[str, Type[InferenceTensor]] = {}


def register_inference_tensor(ty: Type[QuantizedLayoutT]) -> Type[QuantizedLayoutT]:
    """Class decorator which registers a serializable InferenceTensor class.

    This should only be used to decorate concrete implementations that need to
    be loaded by name.
    """
    name = ty.serialized_name()
    existing = REGISTERED_INFERENCE_TENSOR_CLASSES.get(name)
    assert (
        existing is None
    ), f"Duplicate InferenceTensor registration '{name}' ({ty} vs {existing})"
    REGISTERED_INFERENCE_TENSOR_CLASSES[name] = ty
    return ty


########################################################################################
# Primitive tensors
########################################################################################


class PrimitiveTensor(InferenceTensor):
    """An InferenceTensor without any kind of special layout.

    These can be directly operated on as a torch.Tensor.
    """

    @abstractmethod
    def as_torch(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Accesses the raw data as a torch tensor.

        If the tensor is packed in some way, this may bare no resemblance to
        the logical arrangement of the data.
        """
        ...

    def contiguous(self) -> "PrimitiveTensor":
        return self.as_torch().contiguous()

    @property
    def device(self) -> torch.device:
        return self.as_torch().device

    def dim(self) -> int:
        return self.as_torch().dim()

    @property
    def dtype(self) -> torch.dtype:
        return self.as_torch().dtype

    def __setitem__(self, key, value: "AnyTensor"):
        if not isinstance(key, list) and not isinstance(key, tuple):
            key = (key,)

        key = [unbox_tensor(k) if isinstance(k, PrimitiveTensor) else k for k in key]
        self.as_torch()[*key] = unbox_tensor(value)


@register_inference_tensor
class DefaultPrimitiveTensor(PrimitiveTensor):
    """Concrete implementation of a PrimitiveTensor based on a single tensor."""

    def __init__(
        self,
        *,
        data: torch.Tensor,
        name: str = UnnamedTensorName,
    ):
        super().__init__(name=name, shape=list(data.shape))
        assert isinstance(data, torch.Tensor), "data argument must be torch.Tensor"
        self._data = data

    @classmethod
    def serialized_name(cls) -> str:
        return "PrimitiveTensor"

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        try:
            data = raw_tensors[""]
        except KeyError as e:
            raise IOError(f"Missing component tensor") from e
        return cls(name=name, data=data)

    def as_torch(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is not None:
            return self._data.to(dtype)
        return self._data

    @property
    def subtensors(self) -> dict[str, torch.Tensor]:
        return {
            self.name: self._data,
        }

    def get_metadata(self) -> InferenceTensorMetadata:
        return InferenceTensorMetadata(self.serialized_name(), {"": self.name})

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        builder.add_tensor(self.name, self._data)
        return self.get_metadata()

    def _clone_with_subtensors(
        self, new_subtensors: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        return DefaultPrimitiveTensor(name=self.name, data=new_subtensors[self.name])

    def __invert__(self):
        return DefaultPrimitiveTensor(data=~self._data, name=self.name)

    def __getitem__(self, key):
        keys = [key]
        if isinstance(key, tuple) or isinstance(key, list):
            keys = key

        keys = [
            unbox_tensor(key) if isinstance(key, PrimitiveTensor) else key
            for key in keys
        ]
        return self._data[*keys]

    def __repr__(self):
        return f"PrimitiveTensor({self.name}, {self.shape}, {self._data.dtype})"

    def is_deep_equal(self, other: Any, *, compare_name: bool = True) -> bool:
        if not isinstance(other, DefaultPrimitiveTensor):
            return False
        if not self._is_deep_equal(other, compare_name=compare_name):
            return False
        return torch.equal(self.as_torch(), other.as_torch())


########################################################################################
# Quantized tensors
########################################################################################


class QuantizedTensor(InferenceTensor, Generic[QuantizedLayoutT]):
    """An inference tensor that is quantized/packed."""

    def __init__(
        self,
        *,
        shape: list[int],
        layout_type: Type[QuantizedLayout],
        name: str = UnnamedTensorName,
    ):
        super().__init__(name=name, shape=shape)
        self.layout_type = layout_type

    def unpack(self) -> QuantizedLayoutT:
        from amdsharktank import ops

        return ops.unpack(self)

    def to_planar(self) -> "PlanarQuantizedTensor":
        """Converts this QuantizedTensor to a generic planar form.

        This is done for serialization and to materialize unpacking.
        If a subclass cannot be converted to planar form generically like this,
        it should override this method to implement properly or raise
        NotImplementedError.
        """
        return PlanarQuantizedTensor(
            name=self.name, shape=self.shape, layout=self.unpack()
        )

    def get_metadata(self) -> InferenceTensorMetadata:
        return self.to_planar().get_metadata()

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """By default all QuantizedTensors serialize as a generic PlanarQuantizedTensor.

        If this is not desirable, subclasses should override.
        """
        return self.to_planar().add_to_archive(builder)


@register_inference_tensor
class PlanarQuantizedTensor(QuantizedTensor):
    """Generic planar tensor backed by an instantiated QuantizedLayout.

    This is used for materialized, unpacked layouts (i.e. no unpacking
    will be done).
    """

    def __init__(
        self,
        *,
        shape: list[int],
        layout: QuantizedLayout,
        name: str = UnnamedTensorName,
    ):
        super().__init__(name=name, shape=shape, layout_type=type(layout))
        self.layout = layout

    def to_planar(self) -> "PlanarQuantizedTensor":
        # Already planar.
        return self

    @classmethod
    def serialized_name(cls) -> str:
        return "PlanarQuantizedTensor"

    @property
    def subtensors(self) -> dict[str, torch.Tensor]:
        global_name = self.name
        planes = self.layout.planes
        return {f"{global_name}:{k}": v for k, v in planes.items()}

    def _clone_with_subtensors(
        self, new_subtensors: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        # Clone it via layout serialization.
        serialized_name = self.layout.serialized_name()
        global_prefix = f"{self.name}:"
        orig_planes = self.layout.planes
        new_planes = {}
        for plane_name in orig_planes.keys():
            # Planes are stored in the globals dict with the inference
            # tensor's name and colon prepended, so look up that way.
            new_planes[plane_name] = new_subtensors[f"{global_prefix}{plane_name}"]

        # Create a new layout via the serialization adaptor.
        try:
            layout_clazz = REGISTERED_LAYOUT_CLASSES[serialized_name]
        except KeyError:
            raise IOError(
                f"Cannot deserialize PlanarQuantizedTensor because of unregistered layout "
                f"{serialized_name}"
            )
        new_layout = layout_clazz.create(self.shape, self.layout.metadata, new_planes)

        return PlanarQuantizedTensor(
            name=self.name,
            shape=self.shape,
            layout=new_layout,
        )

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        try:
            shape = extra_properties["shape"]
            layout_type_name = extra_properties["layout_type"]
            layout_metadata = extra_properties.get("layout_metadata")
        except KeyError as e:
            raise IOError(f"Missing PlanarQuantizedTensor deserialization prop") from e

        shape = [int(d) for d in shape]
        try:
            layout_clazz = REGISTERED_LAYOUT_CLASSES[layout_type_name]
        except KeyError:
            raise IOError(
                f"Cannot deserialize PlanarQuantizedTensor because of unregistered layout "
                f"{layout_type_name}"
            )

        layout = layout_clazz.create(shape, layout_metadata, raw_tensors)
        return PlanarQuantizedTensor(name=name, shape=shape, layout=layout)

    def get_metadata(self) -> InferenceTensorMetadata:
        root_name = self.name
        layout = self.unpack()
        name_map: dict[str, str] = {}
        for suffix in layout.planes.keys():
            irpa_name = f"{root_name}:{suffix}"
            name_map[suffix] = irpa_name
        extra_properties = {
            "shape": [int(d) for d in self.shape],
            "layout_type": self.layout.serialized_name(),
        }
        layout_metadata = self.layout.metadata
        if layout_metadata is not None:
            extra_properties["layout_metadata"] = layout_metadata
        return InferenceTensorMetadata(
            PlanarQuantizedTensor.serialized_name(),
            name_map,
            extra_properties=extra_properties,
        )

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        meta = self.get_metadata()
        layout = self.unpack()
        for name, irpa_name in meta.raw_tensors.items():
            builder.add_tensor(irpa_name, layout.planes[name])
        return meta

    def __repr__(self):
        return f"PlanarQuantizedTensor({self.name}, {self.shape}, layout={self.layout})"


########################################################################################
# Sharded tensors
########################################################################################


class ShardedTensor(InferenceTensor):
    """A sharded tensor contains a list of tensor-parallel shards, one for each rank.

    The shape of the overall sharded tensor is the un-sharded shape.
    """

    def __init__(
        self,
        *,
        ts: list[torch.Tensor] | list[DefaultPrimitiveTensor] | list[QuantizedTensor],
        shape: list[int],
        shard_dim: int | None,
        name: str = UnnamedTensorName,
        devices: Tuple[int],
    ):
        super().__init__(name=name, shape=shape)
        self.shard_dim = shard_dim
        self._devices = devices

        _shards = []
        for i, t in enumerate(ts):
            if isinstance(t, torch.Tensor):
                t = DefaultPrimitiveTensor(data=t)
            if ".shard." not in t.name:
                t.name = f"{name}.shard.{i}"
            _shards.append(t)
        self._shards: tuple[InferenceTensor, ...] = tuple(_shards)

        for i, shard in enumerate(self._shards):
            assert (
                f".shard.{i}" in shard.name
            ), f"Shard {i} of {name} has name {shard.name}, expected {name}.shard.{i}"

        assert all(
            not isinstance(s, ShardedTensor) for s in self._shards
        ), "ShardedTensor within a shard of a sharded tensor is not supported."

    def __invert__(self):
        return self.clone(ts=[~t for t in self._shards])

    @property
    def device(self) -> torch.device:
        assert all(s.device == self.shards[0].device for s in self.shards), (
            "TODO: figure out what do do if shards are placed on different Torch "
            "devices. This is only relevant for eager execution."
        )
        return self.shards[0].as_torch().device

    @property
    def devices(self) -> Tuple[int]:
        return self._devices

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    @property
    def shards(self) -> tuple[InferenceTensor]:
        """Accesses the underlying shards"""
        return self._shards

    @property
    @abstractmethod
    def is_replicated(self) -> bool:
        """Returns whether the original tensor is replicated.
        If replicated, `shard_dim` does not make sense and is None."""
        ...

    @abstractmethod
    def clone(self, **kwargs) -> "ShardedTensor":
        """
        Create a clone of this tensor with the given properties overridden.
        NOTE: Changing the `devices` will NOT transfer the shards to the corresponding devices.
        """
        ...

    @InferenceTensor.name.setter
    def name(self, name: str):
        super(ShardedTensor, self.__class__).name.__set__(self, name)
        for i, shard in enumerate(self.shards):
            shard.name = f"{name}.shard.{i}"

    def dim(self) -> int:
        return self.shards[0].dim()

    @property
    def dtype(self) -> torch.dtype:
        return self.shards[0].dtype

    @property
    def subtensors(self) -> dict[str, torch.Tensor]:
        _subtensors = {}
        for shard in self.shards:
            for name, subtensor in shard.subtensors.items():
                _subtensors[name] = subtensor
        return _subtensors

    @staticmethod
    def move_shards_to_new_devices(
        shards: Tuple[Tensor | DefaultPrimitiveTensor | QuantizedTensor, ...],
        *,
        old_devices: Tuple[int, ...] | None = None,
        new_devices: Tuple[int, ...],
    ) -> Tuple[Tensor | DefaultPrimitiveTensor | QuantizedTensor, ...]:
        from amdsharktank.ops import (
            transfer_to_logical_device,
            barrier_on_logical_device,
        )

        assert len(shards) == len(
            new_devices
        ), f"Expected {len(shards)} new devices, got {len(new_devices)} instead."

        if old_devices is None:
            return tuple(
                transfer_to_logical_device(shard, new_devices[j])
                for j, shard in enumerate(shards)
            )

        assert len(shards) == len(
            old_devices
        ), f"Expected {len(shards)} old devices, got {len(old_devices)} instead."

        if list(old_devices) == list(new_devices):
            # Exiting early to not create redundant transfers.
            # The redudant transfers slow down compilation and can cause device placement issues.
            return shards

        return tuple(
            (
                transfer_to_logical_device(shard, new_devices[j])
                if new_devices[j] != old_devices[j]
                else barrier_on_logical_device(shard, new_devices[j])
            )
            for j, shard in enumerate(shards)
        )

    def is_deep_equal(self, other: Any, compare_name: bool = True) -> bool:
        if type(self) != type(other):
            return False
        if self.shard_count != other.shard_count or self.shard_dim != other.shard_dim:
            return False
        if not iterables_equal(self.devices, other.devices):
            return False
        if not self._is_deep_equal(other, compare_name=compare_name):
            return False
        return all(
            a.is_deep_equal(b, compare_name=compare_name)
            for a, b in zip(self.shards, other.shards)
        )


@register_inference_tensor
class ShardedTensorBase(ShardedTensor):
    """Sharded tensor which contains tensors.

    The sharded tensors have names with this tensor's name as the stem and
    a suffix of f".shard.{i}" where i is from 0..shard_count-1.
    """

    def __init__(
        self,
        *,
        shard_dim: int | None,
        ts: list[torch.Tensor] | list[DefaultPrimitiveTensor] | list[QuantizedTensor],
        name: str = UnnamedTensorName,
        shape: Optional[list[int]],
        devices: Tuple[int] | None,
    ):
        assert len(ts) > 0
        assert shard_dim is None or (shard_dim >= 0 and len(ts[0].shape) > shard_dim)
        if devices is None:
            devices = tuple(range(len(ts)))
        assert len(ts) == len(devices)
        super().__init__(
            ts=ts, name=name, shape=shape, shard_dim=shard_dim, devices=devices
        )

    @property
    def is_replicated(self) -> bool:
        return False

    @classmethod
    def serialized_name(cls) -> str:
        return cls.__name__

    def get_metadata(self) -> InferenceTensorMetadata:
        extra_properties = {
            "shard_count": len(self._shards),
            "shape": list(self.shape),
        }
        if self.shard_dim is not None:
            extra_properties.update({"shard_dim": self.shard_dim})
        return InferenceTensorMetadata(
            self.serialized_name(),
            {str(i): pt.name for i, pt in enumerate(self._shards)},
            extra_properties=extra_properties,
        )

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        for i, pt in enumerate(self._shards):
            builder.for_rank(i).add_tensor(pt.name, pt._data)
        return self.get_metadata()

    def _clone_with_subtensors(
        self, new_subtensors: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        new_subtensors = new_subtensors.copy()
        # NOTE: Assuming that the type of each shard does not change
        if len(self._shards) == 1:
            ts = [self._shards[0]._clone_with_subtensors(new_subtensors)]
        else:
            all_new_keys = list(new_subtensors.keys())
            ts = []
            for i, shard in enumerate(self._shards):
                shard_i_key = f".shard.{i}"
                matching_keys = [k for k in all_new_keys if shard_i_key in k]
                new_sub_globals = {k: new_subtensors.pop(k) for k in matching_keys}
                ts.append(shard._clone_with_subtensors(new_sub_globals))
        if self.shard_dim is None:
            return self.__class__(
                name=self.name,
                shape=self.shape,
                ts=ts,
                devices=self.devices,
            )
        else:
            return self.__class__(
                name=self.name,
                shape=self.shape,
                shard_dim=self.shard_dim,
                ts=ts,
                devices=self.devices,
            )

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        shard_count = int(extra_properties["shard_count"])
        shape = list(extra_properties["shape"])
        shard_dim = (
            int(extra_properties["shard_dim"])
            if "shard_dim" in extra_properties
            else None
        )
        ts = []
        for i in range(shard_count):
            t_name = str(i)
            try:
                t = raw_tensors[t_name]
                ts.append(t)
                # TODO: this should be changed to tracked device affinity
                DeviceTensorTrait(i).set(t)
            except KeyError as e:
                raise IOError(
                    f"Missing component tensor '{t_name}' in {raw_tensors.keys()}"
                ) from e
        if shard_dim is None:
            return cls(name=name, shape=shape, ts=ts)
        else:
            return cls(name=name, shape=shape, ts=ts, shard_dim=shard_dim)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.name}, {self.shape}, "
            + ("" if self.shard_dim is None else f"shard_dim={self.shard_dim}, ")
            + f"shard_count={len(self._shards)} "
            f"of {self.shards[0].shape})"
        )


def _is_tuple_of_integral_numbers(x) -> bool:
    if not isinstance(x, tuple):
        return False
    return all(isinstance(el, Integral) for el in x)


def _is_collection_of_integral_numbers(x) -> bool:
    if not isinstance(x, Collection):
        return False
    return all(isinstance(el, Integral) for el in x)


def _is_full_slice(s: slice, dim_size: int) -> bool:
    return (
        (s.start is None or s.start == 0)
        and (s.stop is None or s.stop == dim_size)
        and (s.step is None or s.step == 1)
    )


def _resolve_ellipsis_in_slicing(key: Tuple[Any], shape: Tuple[int]) -> Tuple[Any]:
    """Example:
    key = [1:2, ..., 0]
    shape = [2, 3, 4, 5, 6]
    Returns:
    [1:2, :, :, :, 0]"""
    num_ellipsis = len([k for k in key if k == Ellipsis])
    assert num_ellipsis <= 1, "Only one Ellipses is allowed."
    if num_ellipsis <= 0:
        return key
    assert len(key) <= len(
        shape
    ), "Inserting trailing singleton dimensions is not supported."
    dim = 0
    res = []
    for k in key:
        if k == Ellipsis:
            ellipsis_num_dims = len(shape) - len(key) + 1
            res.extend([slice(None)] * ellipsis_num_dims)
            dim += ellipsis_num_dims
        else:
            dim += 1
            res.append(k)
    return tuple(res)


# TODO: rename to SplitTensor as now the shards can be any InferenceTensor.
@register_inference_tensor
class SplitPrimitiveTensor(ShardedTensorBase):
    """Sharded tensor split along a dimension into primitive tensors.

    The sharded tensors have names with this tensor's name as the stem and
    a suffix of f".shard.{i}" where i is from 0..shard_count-1.
    """

    def __init__(
        self,
        *,
        shard_dim: int,
        ts: list[torch.Tensor | QuantizedTensor] | torch.Tensor | QuantizedTensor,
        shard_count: None | int = None,
        name: str = UnnamedTensorName,
        shape: Optional[list[int]] = None,
        devices: Tuple[int] | None = None,
    ):
        """
        If `ts` is a list of tensors, it is interpreted as the shards.
        Then `shard_count` must be None.
        If `ts` is a tensor then `shard_count` must be provided and it,
        will be split along dimension `shard_dim` into `shard_count`
        number of pieces.
        """
        if devices is None:
            num_shards = len(ts) if isinstance(ts, Sequence) else shard_count
            devices = tuple(range(num_shards))

        if not isinstance(ts, Sequence):
            from amdsharktank.ops import transfer_to_logical_device

            assert shard_count is not None
            assert (
                shard_count > 1
            ), "SplitTensor must have at least 2 shards. Use ReplicatedTensor for 1 shard."
            assert (
                ts.shape[shard_dim] >= shard_count
            ), f"Cannot split dimension {shard_dim} of size {ts.shape[shard_dim]} into {shard_count} shards"
            ts = ts.split(ceildiv(ts.shape[shard_dim], shard_count), dim=shard_dim)
            ts = [transfer_to_logical_device(t, devices[i]) for i, t in enumerate(ts)]
            assert len(ts) == shard_count
            shard_count = None

        assert shard_count is None
        assert (
            len(ts) > 1
        ), "SplitTensor must have at least 2 shards. Use ReplicatedTensor for 1 shard."
        first_shape = ts[0].shape
        assert len(first_shape) > shard_dim
        expected_shape = list(first_shape)
        expected_shape[shard_dim] = sum([t.shape[shard_dim] for t in ts])
        if shape is not None:
            shape = list(shape)
            assert expected_shape == shape
        else:
            shape = expected_shape

        # Assert the shapes.
        for i, t in enumerate(ts):
            t_shape = list(t.shape)
            assert len(shape) == len(
                t_shape
            ), f"Shape size mismatch tensor shard {i} with shape {t.shape}. Expected shape size {len(shape)}. Got {len(t_shape)}."
            assert all(
                s == t for i, (s, t) in enumerate(zip(shape, t_shape)) if i != shard_dim
            ), f"Shape mismatch for non-split dimension for tensor shard {i} with shape {t.shape}"

        super().__init__(
            name=name,
            ts=ts,
            shape=shape,
            shard_dim=shard_dim,
            devices=devices,
        )

    def clone(self, **kwargs) -> "SplitPrimitiveTensor":
        kwargs["name"] = kwargs.get("name", self.name)
        kwargs["devices"] = kwargs.get("devices", self.devices)
        kwargs["shape"] = kwargs.get("shape", self.shape)
        kwargs["shard_dim"] = kwargs.get("shard_dim", self.shard_dim)

        if "ts" in kwargs:
            # Only override shard_count if ts is a tensor.
            if isinstance(kwargs["ts"], torch.Tensor):
                kwargs["shard_count"] = kwargs.get("shard_count", self.shard_count)
        else:
            kwargs["ts"] = self.shards
        return SplitPrimitiveTensor(**kwargs)

    def _is_slicing_split_dim(self, key):
        if isinstance(
            key,
            (
                slice,
                Integral,
            ),
        ):
            return self._is_slicing_split_dim([key])
        if _is_collection_of_integral_numbers(key):
            if isinstance(key, tuple):
                # Index per dimension.
                return self.shard_dim < len(key)
            else:
                # Any other collection is a indexing only dimension 0.
                return self.shard_dim == 0
        if len(key) <= self.shard_dim:
            return False
        if not isinstance(key[self.shard_dim], slice):
            return True
        return not _is_full_slice(key[self.shard_dim], self.shape[self.shard_dim])

    def _get_shard_slice(self, key):
        if isinstance(
            key,
            (
                slice,
                Integral,
            ),
        ):
            return self._get_shard_slice([key])
        if _is_collection_of_integral_numbers(key) and not isinstance(key, tuple):
            # Indexing of dimension 0 only.
            return key
        if len(key) <= self.shard_count:
            return key
        new_key = list(key)

        if self.shard_dim < len(new_key):
            new_key[self.shard_dim] = slice(None)
        return new_key

    def __getitem__(self, key):
        # TODO: implement all cases.
        if not isinstance(key, Sequence):
            key = (key,)
        key = _resolve_ellipsis_in_slicing(key, self.shape)
        if self._is_slicing_split_dim(key):
            raise NotImplementedError(
                f"Slicing of the split dimension {self.shard_dim} is not supported."
            )
        new_key = self._get_shard_slice(key)

        shards = []
        for i, shard in enumerate(self.shards):
            shard_keys = [
                k.shards[i] if isinstance(k, ReplicatedTensor) else k for k in new_key
            ]
            shards.append(shard[*shard_keys])

        shard_dim = self.shard_dim
        for i in range(min(shard_dim, len(key))):
            if isinstance(key[i], Number) and key[i] >= 0:
                # Rank reduction dimension before the split dim.
                shard_dim -= 1

        return SplitPrimitiveTensor(
            ts=shards, shard_dim=shard_dim, devices=self.devices
        )

    def __setitem__(self, key, value):
        assert isinstance(value, SplitPrimitiveTensor)
        assert self.shard_count == value.shard_count
        if not isinstance(key, Sequence):
            key = (key,)
        key = _resolve_ellipsis_in_slicing(key, self.shape)
        if self._is_slicing_split_dim(key):
            raise NotImplementedError(
                f"Slicing of the split dimension {self.shard_dim} is not supported."
            )
        for i, (shard, value_shard) in enumerate(zip(self.shards, value.shards)):
            shard_keys = [
                k.shards[i] if isinstance(k, ReplicatedTensor) else k for k in key
            ]
            shard[*shard_keys] = unbox_tensor(value_shard)


@register_inference_tensor
class ReplicatedTensor(ShardedTensor):
    """A tensor that is replicated across all shards."""

    def __init__(
        self,
        *,
        ts: Union[list["AnyTensor"], "AnyTensor"],
        shard_count: None | int = None,
        name: str = UnnamedTensorName,
        devices: Tuple[int] | None = None,
    ):
        """
        If `ts` is a list of tensors, it is interpreted as the shards.
        Then `shard_count` must be None.
        If `ts` is a tensor then `shard_count` must be provided and it,
        will be replicated that many times.
        """
        if devices is None:
            num_shards = len(ts) if isinstance(ts, list) else shard_count
            devices = tuple(range(num_shards))

        if not isinstance(ts, Sequence):
            assert shard_count is not None
            from amdsharktank.ops import transfer_to_logical_device

            ts = [
                transfer_to_logical_device(ts, devices[i]) for i in range(shard_count)
            ]
            for i in range(len(ts)):
                if isinstance(ts[i], InferenceTensor):
                    ts[i].name = f"{name}.shard.{i}"
            shard_count = None

        assert shard_count is None
        assert len(ts) > 0
        assert len(ts) == len(devices)
        for shard in ts[1:]:
            assert all(s_0 == s_i for (s_0, s_i) in zip(ts[0].shape, shard.shape))

        super().__init__(
            ts=ts, name=name, shape=list(ts[0].shape), shard_dim=None, devices=devices
        )

    def clone(self, **kwargs) -> "ReplicatedTensor":
        kwargs["ts"] = kwargs.get("ts", self.shards)
        kwargs["name"] = kwargs.get("name", self.name)
        kwargs["devices"] = kwargs.get("devices", self.devices)

        if "ts" in kwargs:
            # Only override shard_count if ts is a tensor.
            if isinstance(kwargs["ts"], torch.Tensor):
                kwargs["shard_count"] = kwargs.get("shard_count", self.shard_count)
        else:
            kwargs["ts"] = self.shards
        return ReplicatedTensor(**kwargs)

    @property
    def is_replicated(self) -> bool:
        return True

    @classmethod
    def serialized_name(cls) -> str:
        return "ReplicatedTensor"

    def get_metadata(self) -> InferenceTensorMetadata:
        return InferenceTensorMetadata(
            self.serialized_name(),
            {"": self._shards[0].name},
            extra_properties={
                "shard_count": len(self._shards),
            },
        )

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        builder.for_rank(0).add_tensor(self._shards[0].name, self._shards[0]._data)
        return self.get_metadata()

    def _clone_with_subtensors(
        self, new_subtensors: dict[str, torch.Tensor]
    ) -> "ReplicatedTensor":
        new_subtensors = new_subtensors.copy()
        # NOTE: Assuming that the type of each shard does not change
        if len(self._shards) == 1:
            ts = [self._shards[0]._clone_with_subtensors(new_subtensors)]
        else:
            all_new_keys = list(new_subtensors.keys())
            ts = []
            for i, shard in enumerate(self._shards):
                shard_i_key = f".shard.{i}"
                matching_keys = [k for k in all_new_keys if shard_i_key in k]
                new_sub_globals = {k: new_subtensors.pop(k) for k in matching_keys}
                ts.append(shard._clone_with_subtensors(new_sub_globals))
        return ReplicatedTensor(
            name=self.name,
            ts=ts,
            devices=self.devices,
        )

    @classmethod
    def create(
        cls,
        name: str,
        raw_tensors: dict[str, torch.Tensor],
        extra_properties: dict[str, Any],
    ) -> "InferenceTensor":
        shard_count = int(extra_properties["shard_count"])
        try:
            # We have to do this to avoid exporting as part of the `mlir` blob:
            t = raw_tensors[""]
            ts = [raw_tensors[""]]
            for i in range(1, shard_count):
                nt = deepcopy(t)
                ts.append(nt)

            # TODO This should be changed to assigned affinities
            for i in range(shard_count):
                DeviceTensorTrait(i).set(ts[i])

        except KeyError as e:
            raise IOError(f"Missing component tensor '' in {raw_tensors.keys()}") from e
        return cls(name=name, ts=ts)

    def __getitem__(self, key):
        keys = [key]
        if isinstance(key, tuple) or isinstance(key, list):
            keys = key

        shards = []
        for i, shard in enumerate(self.shards):
            shard_keys = []
            for k in keys:
                if isinstance(k, ReplicatedTensor):
                    shard_keys.append(k.shards[i])
                else:
                    shard_keys.append(k)
            shards.append(shard[*shard_keys])
        return ReplicatedTensor(ts=shards, devices=self.devices)

    def __repr__(self):
        return (
            f"ReplicatedTensor({self.name}, {self.shape}, "
            f"shard_count={len(self._shards)} "
            f"of {self.shards[0].shape})"
        )


@register_inference_tensor
class UnreducedTensor(ShardedTensorBase):
    """Sharded tensor which contains primitive tensors.
    To obtain the actual tensor a sum-reduction over the shards must be performed.
    """

    def __init__(
        self,
        *,
        ts: list[torch.Tensor],
        name: str = UnnamedTensorName,
        shape: Optional[list[int]] = None,
        devices: Tuple[int] | None = None,
    ):
        assert len(ts) > 0
        shape = list(ts[0].shape if shape is None else shape)
        assert all(shape == list(t.shape) for t in ts)
        super().__init__(
            name=name,
            ts=ts,
            shape=shape,
            shard_dim=None,
            devices=devices,
        )

    def clone(self, **kwargs) -> "UnreducedTensor":
        kwargs["ts"] = kwargs.get("ts", self.shards)
        kwargs["name"] = kwargs.get("name", self.name)
        kwargs["shape"] = kwargs.get("shape", self.shape)
        kwargs["devices"] = kwargs.get("devices", self.devices)
        return UnreducedTensor(**kwargs)


def is_any_tensor(x: Any) -> bool:
    return isinstance(x, (InferenceTensor, torch.Tensor))


def flatten_tensor_tree(
    tree: tree_utils.Tree,
) -> Iterable[torch.Tensor | InferenceTensor]:
    """Flatten up to our tensor types."""
    return tree_utils.flatten(
        tree,
        is_leaf=lambda x: isinstance(
            x,
            (
                torch.Tensor,
                InferenceTensor,
            ),
        ),
    )


def unbox_tensor(t: Any) -> Tensor:
    """Unboxes a value that can be isomorphically interpreted as a Tensor."""
    if isinstance(t, Tensor):
        return t
    elif isinstance(t, PrimitiveTensor):
        return t.as_torch()
    elif isinstance(t, QuantizedTensor):
        return t.unpack().dequant()
    elif isinstance(t, ShardedTensor):
        from .. import ops

        return unbox_tensor(ops.unshard(t))
    raise ValueError(f"Expected a Tensor or PrimitiveTensor but got {type(t)}")


########################################################################################
# Serialization helpers
########################################################################################


def dtype_to_serialized_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_TO_NAME[dtype]
    except KeyError as e:
        raise KeyError(
            f"Missing mapping for dtype {dtype}. Please add to the _NAME_TO_DTYPE dict"
        ) from e


def dtype_to_serialized_short_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_TO_SHORT_NAME[dtype]
    except KeyError as e:
        raise KeyError(
            f"Missing mapping for dtype {dtype}. Please add to the _SHORT_NAME_TO_DTYPE dict"
        ) from e


def serialized_name_to_dtype(dtype_name: str) -> torch.dtype:
    try:
        return _NAME_TO_DTYPE[dtype_name]
    except KeyError as e:
        raise KeyError(
            f"Missing mapping for dtype '{dtype_name}'. Please add to the _NAME_TO_DTYPE dict"
        ) from e


def serialized_short_name_to_dtype(dtype_name: str) -> torch.dtype:
    try:
        return _SHORT_NAME_TO_DTYPE[dtype_name]
    except KeyError as e:
        raise KeyError(
            f"Missing mapping for dtype '{dtype_name}'. Please add to the _SHORT_NAME_TO_DTYPE dict"
        ) from e


_NAME_TO_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "float8_e4m3fnuz": torch.float8_e4m3fnuz,
}


def _maybe_dtype(*names: str):
    for name in names:
        try:
            cls = getattr(torch, name)
        except AttributeError:
            pass
        else:
            _NAME_TO_DTYPE[name] = cls


_maybe_dtype(
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
)

_DTYPE_TO_NAME: dict[torch.dtype, str] = {v: k for k, v in _NAME_TO_DTYPE.items()}

_SHORT_NAME_TO_DTYPE: dict[str, torch.dtype] = {
    "f32": torch.float32,
    "f64": torch.float64,
    "c64": torch.complex64,
    "c128": torch.complex128,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "ui8": torch.uint8,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "b": torch.bool,
    "f8_e4m3fnuz": torch.float8_e4m3fnuz,
}

_DTYPE_TO_SHORT_NAME: dict[torch.dtype, str] = {
    v: k for k, v in _SHORT_NAME_TO_DTYPE.items()
}

AnyTensor = Union[torch.Tensor, InferenceTensor]
AnyTensorTree = (
    Mapping[Any, Union[Any, "AnyTensorTree"]]
    | Iterable[Union[Any, "AnyTensorTree"]]
    | Any
)

########################################################################################
# Tensor types registration with PyTorch.
# This enables our tensor types to be part of function signatures during exporting.
########################################################################################


def flatten_default_primitive_tensor(
    t: DefaultPrimitiveTensor,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    return [t.as_torch()], {"name": t.name}


def unflatten_defult_primitive_tensor(
    values: Iterable[Any], ctx: torch.utils._pytree.Context
) -> DefaultPrimitiveTensor:
    values_as_list = list(values)
    return DefaultPrimitiveTensor(data=values_as_list[0], name=ctx["name"])


def flatten_with_keys_default_primitive_tensor(t: DefaultPrimitiveTensor):
    values, context = flatten_default_primitive_tensor(t)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


register_pytree_node(
    DefaultPrimitiveTensor,
    flatten_fn=flatten_default_primitive_tensor,
    unflatten_fn=unflatten_defult_primitive_tensor,
    flatten_with_keys_fn=flatten_with_keys_default_primitive_tensor,
)


def flatten_split_primitive_tensor(
    t: SplitPrimitiveTensor,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    return t.shards, {
        "name": t.name,
        "shard_dim": t.shard_dim,
        "devices": t.devices,
    }


def unflatten_split_primitive_tensor(
    values: Iterable[Any], ctx: torch.utils._pytree.Context
) -> SplitPrimitiveTensor:
    return SplitPrimitiveTensor(
        shard_dim=ctx["shard_dim"],
        ts=list(values),
        name=ctx["name"],
        devices=ctx["devices"],
    )


def flatten_with_keys_split_primitive_tensor(t: SplitPrimitiveTensor):
    values, context = flatten_split_primitive_tensor(t)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


register_pytree_node(
    SplitPrimitiveTensor,
    flatten_fn=flatten_split_primitive_tensor,
    unflatten_fn=unflatten_split_primitive_tensor,
    flatten_with_keys_fn=flatten_with_keys_split_primitive_tensor,
)


def flatten_replicated_tensor(
    t: ReplicatedTensor,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    return list(t.shards), {"name": t.name, "devices": t.devices}


def unflatten_replicated_tensor(
    values: Iterable[Any], ctx: torch.utils._pytree.Context
) -> ReplicatedTensor:
    return ReplicatedTensor(ts=list(values), name=ctx["name"], devices=ctx["devices"])


def flatten_with_keys_replicated_tensor(t: ReplicatedTensor):
    values, context = flatten_replicated_tensor(t)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


register_pytree_node(
    ReplicatedTensor,
    flatten_fn=flatten_replicated_tensor,
    unflatten_fn=unflatten_replicated_tensor,
    flatten_with_keys_fn=flatten_with_keys_replicated_tensor,
)


def flatten_unreduced_tensor(
    t: UnreducedTensor,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    return list(t.shards), {"name": t.name, "devices": t.devices}


def unflatten_unreduced_tensor(
    values: Iterable[Any], ctx: torch.utils._pytree.Context
) -> UnreducedTensor:
    return UnreducedTensor(ts=list(values), name=ctx["name"], devices=ctx["devices"])


def flatten_with_keys_unreduced_tensor(t: UnreducedTensor):
    values, context = flatten_unreduced_tensor(t)
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


register_pytree_node(
    UnreducedTensor,
    flatten_fn=flatten_unreduced_tensor,
    unflatten_fn=unflatten_unreduced_tensor,
    flatten_with_keys_fn=flatten_with_keys_unreduced_tensor,
)


def torch_tree_flatten(tree: tree_utils.Tree):
    """Flatten a tree of tensors the same way they will be flattened during torch.export.export
    if they are arguments or results of a function signature."""
    return torch.utils._pytree.tree_flatten(tree=tree)
