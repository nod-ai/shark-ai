# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
)
from copy import deepcopy
from collections.abc import Collection, Sequence
from numbers import Integral, Number
import os

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
import torch._subclasses.functional_tensor
from torch.utils._pytree import register_pytree_node, SequenceKey
import torch.utils._pytree
from ..utils.math import ceildiv
from iree.turbine.aot import (
    DeviceTensorTrait,
    ExternalTensorTrait,
)
from ..utils import tree as tree_utils

from ..utils.io import ShardedArchiveBuilder

__all__ = [
    "AnyTensor",
    "DefaultPrimitiveTensor",
    "dtype_to_serialized_name",
    "dtype_to_serialized_short_name",
    "flatten_tensor_tree",
    "InferenceTensor",
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
    "UnreducedTensor",
]

if (
    "SHARKTANK_OVERRIDE_TORCH_TENSOR_REPR" in os.environ
    and os.environ["SHARKTANK_OVERRIDE_TORCH_TENSOR_REPR"] != "0"
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
        ...

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
        """
        ...

    @property
    def metadata(self) -> Optional[dict[str, MetaDataValueType]]:
        """Additional metadata needed to reconstruct a layout."""
        return None


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
    def globals(self) -> dict[str, torch.Tensor]:
        """Returns a mapping of global name to root tensor.

        The primary accessors on an InferenceTensor access the root tensors in
        the global set, all of which in a root Theta must have unique names.
        """
        ...

    @abstractmethod
    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        ...

    def is_deep_equal(self, other: Any) -> bool:
        """Deep equality including metadata and exact equality of tensor elements.
        It is a representational equality."""
        raise NotImplementedError()

    def transform_globals(
        self, *transforms: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    ) -> "InferenceTensor":
        """Appplies transformation functions to the InferenceTensors backing
        globals.

        Each transformation must produce a new dict of a form that the subclass
        can handle. Practically, this means that placement and layout related
        changes are always allowed, while more invasive changes (like dtype)
        are more case by case.

        Returns a new InferenceTensor, mutated.
        """
        prev_globals = self.globals
        for transform in transforms:
            next_globals = transform(prev_globals)
            # Copy any metadata from prior to next.
            for k, prev_t in prev_globals.items():
                new_t = next_globals.get(k)
                if new_t is None:
                    continue
                if new_t is not prev_t:
                    ext_trait = ExternalTensorTrait.get(prev_t)
                    if ext_trait is not None:
                        ext_trait.set(new_t)
            prev_globals = next_globals
        return self._clone_with_globals(prev_globals)

    def to(
        self,
        *,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "InferenceTensor":
        # TODO: reconcile with ops.to(...) and torch.Tensor.to(...).
        # Do we always want to clone with globals?
        # This makes our type inconsistent with torch tensors.
        # If we use this to transform a theta we want to change the theta.
        # If we want to use this in a computation we don't want to change the theta.
        return self.transform_globals(
            lambda d: {k: t.to(device=device) for k, t in d.items()}
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        raise NotImplementedError(
            f"InferenceTensor {type(self)} does not implement _clone_with_globals"
        )

    @property
    def T(self) -> "InferenceTensor":
        from ..ops import permute

        # Reverse the dimension range.
        rank = len(self.shape)
        assert rank == 2, "T will be deprecated in torch for non-2D tensors"
        dims = [rank - 1 - i for i in range(rank)]

        return permute(self, dims=dims)

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError()

    def expand(self, *args: Union[List[List[int]], List[int]]) -> "AnyTensor":
        from ..ops import expand

        if all(isinstance(a, int) for a in args):
            shape = args
        else:
            assert len(args) == 1
            shape = args[0]
        return expand(self, shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "AnyTensor":
        from ..ops import flatten

        return flatten(self, start_dim, end_dim)

    def index_copy_(
        self, dim: int, index: "AnyTensor", tensor: "AnyTensor"
    ) -> "InferenceTensor":
        from ..ops import index_copy_

        return index_copy_(self, dim, index, tensor)

    def index_put_(
        self, indices: Tuple["AnyTensor"], values: "AnyTensor"
    ) -> "InferenceTensor":
        from ..ops import index_put_

        return index_put_(self, indices, values)

    def index_select(
        self,
        dim: int,
        index: "AnyTensor",
    ) -> "InferenceTensor":
        from ..ops import index_select

        return index_select(self, dim, index)

    def mean(
        self,
        dim: Union[int, List[int]],
        keepdim: bool = False,
        *,
        dtype: torch.dtype = None,
    ) -> "AnyTensor":
        from ..ops import mean

        return mean(self, dim, keepdim, dtype=None)

    def pow(self, exponent: Union["AnyTensor", Number]) -> "AnyTensor":
        from ..ops import elementwise

        return elementwise(torch.pow, self, exponent)

    def repeat(self, *sizes: List[int]) -> "AnyTensor":
        from ..ops import repeat

        return repeat(self, *sizes)

    def reshape(self, *args: Union[List[List[int]], List[int]]) -> "AnyTensor":
        from ..ops import reshape

        if all(isinstance(a, int) for a in args):
            shape = args
        else:
            assert len(args) == 1
            shape = args[0]
        return reshape(self, shape)

    def size(self, dim: Optional[int] = None) -> tuple[int]:
        if dim is None:
            return tuple(self.shape)
        return self.shape[dim]

    def squeeze(self, dim: Optional[int] = None) -> "AnyTensor":
        from ..ops import squeeze

        return squeeze(self, dim)

    def squeeze(self, dim: Optional[int] = None) -> "AnyTensor":
        from ..ops import squeeze

        return squeeze(self, dim)

    def transpose(self, dim0: int, dim1: int) -> "AnyTensor":
        from ..ops import transpose

        return transpose(self, dim0, dim1)

    def unflatten(self, dim: int, sizes: Tuple[int]) -> "AnyTensor":
        from ..ops import unflatten

        return unflatten(self, dim, sizes)

    def unsqueeze(self, dim: int) -> "AnyTensor":
        from ..ops import unsqueeze

        return unsqueeze(self, dim)

    def view(self, *args: Union[List[List[int]], List[int]]) -> "AnyTensor":
        from ..ops import view

        if all(isinstance(a, int) or isinstance(a, torch.SymInt) for a in args):
            shape = args
        else:
            assert len(args) == 1
            shape = args[0]
        return view(self, shape)

    def __add__(self, rhs):
        from ..ops import elementwise

        return elementwise(torch.add, self, rhs)

    def __radd__(self, lhs):
        # Assumes commutative addition due to torch elementwise ops not handling
        # numbers on the lhs.
        return self.__add__(lhs)

    def __mod__(self, rhs):
        from ..ops import elementwise

        return elementwise(torch.remainder, self, rhs)

    def __mul__(self, rhs):
        from ..ops import elementwise

        return elementwise(torch.mul, self, rhs)

    def __rmul__(self, lhs):
        # Assumes commutative multiplication due to torch elementwise ops not handling
        # numbers on the lhs.
        return self.__mul__(lhs)

    def __truediv__(self, rhs):
        from ..ops import elementwise

        return elementwise(torch.true_divide, self, rhs)

    def __floordiv__(self, rhs):
        from ..ops import elementwise

        return elementwise(torch.floor_divide, self, rhs)

    def __getitem__(self, key):
        from ..ops import get_index

        return get_index(self, key)


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
    def globals(self) -> dict[str, torch.Tensor]:
        return {
            self.name: self._data,
        }

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        builder.add_tensor(self.name, self._data)
        return InferenceTensorMetadata(self.serialized_name(), {"": self.name})

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        return DefaultPrimitiveTensor(name=self.name, data=new_globals[self.name])

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

    def is_deep_equal(self, other: Any) -> bool:
        if not isinstance(other, DefaultPrimitiveTensor):
            return False
        if self.shape != other.shape or self.name != other.name:
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

    @abstractmethod
    def unpack(self) -> QuantizedLayoutT:
        ...

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

    def unpack(self) -> QuantizedLayout:
        return self.layout

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        global_name = self.name
        planes = self.layout.planes
        return {f"{global_name}:{k}": v for k, v in planes.items()}

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        # Clone it via layout serialization.
        serialized_name = self.layout.serialized_name()
        global_prefix = f"{self.name}:"
        orig_planes = self.layout.planes
        new_planes = {}
        for plane_name in orig_planes.keys():
            # Planes are stored in the globals dict with the inference
            # tensor's name and colon prepended, so look up that way.
            new_planes[plane_name] = new_globals[f"{global_prefix}{plane_name}"]

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

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        """Adds this tensor to the global archive."""
        root_name = self.name
        layout = self.unpack()
        name_map: dict[str, str] = {}
        for suffix, plane in layout.planes.items():
            irpa_name = f"{root_name}:{suffix}"
            builder.add_tensor(irpa_name, plane)
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

    def __repr__(self):
        return f"PlanarQuantized({self.name}, {self.shape}, layout={self.layout})"


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
        ts: list[torch.Tensor] | list[DefaultPrimitiveTensor],
        shape: list[int],
        shard_dim: int | None,
        name: str = UnnamedTensorName,
        devices: Tuple[int],
    ):
        super().__init__(name=name, shape=shape)
        self.shard_dim = shard_dim
        self._devices = devices
        self._shards: tuple[DefaultPrimitiveTensor] = tuple(
            DefaultPrimitiveTensor(
                name=f"{name}.shard.{i}",
                data=t,
            )
            if isinstance(t, torch.Tensor)
            else t
            for i, t in enumerate(ts)
        )

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

    @property
    def dtype(self) -> torch.dtype:
        return self.shards[0].dtype

    @staticmethod
    def move_shards_to_new_devices(
        shards: Tuple[torch.Tensor | DefaultPrimitiveTensor, ...],
        *,
        old_devices: Tuple[int, ...],
        new_devices: Tuple[int, ...],
    ) -> Tuple[DefaultPrimitiveTensor, ...]:
        from ..ops import transfer_to_logical_device, barrier_on_logical_device

        new_shard_tensors = tuple(
            (
                transfer_to_logical_device(shard, new_devices[j])
                if new_devices[j] != old_devices[j]
                else barrier_on_logical_device(shard, new_devices[j])
            )
            for j, shard in enumerate(shards)
        )
        return tuple(
            DefaultPrimitiveTensor(name=orig_dpt.name, data=new_shard_tensor)
            for orig_dpt, new_shard_tensor in zip(shards, new_shard_tensors)
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
        ts: list[torch.Tensor],
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

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {pt.name: pt._data for pt in self._shards}

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        for i, pt in enumerate(self._shards):
            builder.for_rank(i).add_tensor(pt.name, pt._data)
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

    @classmethod
    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        ts = []
        for k in self.globals.keys():
            ts.append(new_globals[ts[k]])
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

    def is_deep_equal(self, other: Any) -> bool:
        if type(self) != type(other):
            return False
        if (
            self.shard_count != other.shard_count
            or self.shard_dim != other.shard_dim
            or self.name != other.name
            or self.shape != other.shape
        ):
            return False
        return all(a.is_deep_equal(b) for a, b in zip(self.shards, other.shards))


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
        ts: list[torch.Tensor] | torch.Tensor,
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
            num_shards = shard_count if isinstance(ts, torch.Tensor) else len(ts)
            devices = tuple(range(num_shards))

        if isinstance(ts, torch.Tensor):
            from ..ops import transfer_to_logical_device

            assert shard_count is not None
            assert (
                shard_count > 1
            ), "SplitTensor must have at least 2 shards. Use ReplicatedTensor for 1 shard."
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
        ts: list[torch.Tensor] | torch.Tensor,
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
            num_shards = shard_count if isinstance(ts, torch.Tensor) else len(ts)
            devices = tuple(range(num_shards))

        if isinstance(ts, torch.Tensor):
            assert shard_count is not None
            from ..ops import transfer_to_logical_device

            ts = [
                transfer_to_logical_device(ts, devices[i]) for i in range(shard_count)
            ]
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

    @property
    def globals(self) -> dict[str, torch.Tensor]:
        return {pt.name: pt._data for pt in self._shards}

    def add_to_archive(self, builder: ShardedArchiveBuilder) -> InferenceTensorMetadata:
        builder.for_rank(0).add_tensor(self.name, self._shards[0]._data)
        return InferenceTensorMetadata(
            self.serialized_name(),
            {"": self.name},
            extra_properties={
                "shard_count": len(self._shards),
            },
        )

    def _clone_with_globals(
        self, new_globals: dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        ts = []
        for k in self.globals.keys():
            ts.append(new_globals[ts[k]])
        return ReplicatedTensor(
            name=self.name,
            shape=self.shape,
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

    def is_deep_equal(self, other: Any) -> bool:
        if not isinstance(other, ReplicatedTensor):
            return False
        if (
            self.shard_count != other.shard_count
            or self.name != other.name
            or self.shape != other.shape
        ):
            return False
        if self.shard_count == 0:
            return True
        return self.shards[0].is_deep_equal(other.shards[0])


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
