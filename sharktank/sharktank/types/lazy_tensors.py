# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import abstractmethod
from typing import Dict, Any
import torch

from .tensors import InferenceTensor, InferenceTensorMetadata, register_inference_tensor


class AbstractLazyTensor(InferenceTensor):
    """Base class for tensors that have been reindexed/transformed"""

    def __init__(
        self, base_tensor: InferenceTensor, shape: list[int], name: str = None
    ):
        self._base_tensor = base_tensor
        super().__init__(name=name or base_tensor.name, shape=shape)

    @abstractmethod
    def evaluate(self) -> InferenceTensor:
        """Evaluate the transformation and return result"""
        pass

    @property
    def base_tensor(self) -> InferenceTensor:
        """Return the base tensor"""
        return self._base_tensor

    @property
    def subtensors(self) -> Dict[str, torch.Tensor]:
        """Return the subtensors from the base tensor"""
        return self._base_tensor.subtensors

    def as_torch(self) -> torch.Tensor:
        """Return the transformed tensor as torch.Tensor"""
        return self.evaluate().as_torch()

    def _clone_with_subtensors(
        self, new_subtensors: Dict[str, torch.Tensor]
    ) -> "InferenceTensor":
        """Clone with new subtensors"""
        return self._base_tensor._clone_with_subtensors(new_subtensors)


# Currently this doesn't do anything other than act as a sentinel type.
# I will most likely massively change this.
@register_inference_tensor
class PermutedTensor(AbstractLazyTensor):
    """Tensor that has been permuted with specific dimensions"""

    def __init__(
        self, base_tensor: InferenceTensor, permute_dims: tuple, name: str = None
    ):
        self.permute_dims = permute_dims  # Store for metadata/serialization
        # Calculate the new shape after permutation
        base_shape = list(base_tensor.shape)
        new_shape = [base_shape[i] for i in permute_dims]
        super().__init__(base_tensor, new_shape, name)

    def evaluate(self) -> InferenceTensor:
        """Apply permutation and return transformed tensor"""
        return self._base_tensor.permute(*self.permute_dims)

    @classmethod
    def serialized_name(cls) -> str:
        return "PermutedTensor"

    def get_metadata(self) -> InferenceTensorMetadata:
        """Include permute_dims in metadata so we know it's shuffled"""
        base_metadata = self._base_tensor.get_metadata()

        extra_properties = base_metadata.extra_properties or {}
        extra_properties.update(
            {
                "permute_dims": list(self.permute_dims),
                "is_shuffled": True,
                "original_type": base_metadata.type_name,
            }
        )

        return InferenceTensorMetadata(
            type_name=self.serialized_name(),
            raw_tensors=base_metadata.raw_tensors,
            extra_properties=extra_properties,
        )

    @classmethod
    def create_deserialized(
        cls,
        name: str,
        raw_tensors: Dict[str, torch.Tensor],
        extra_properties: Dict[str, Any],
    ) -> "InferenceTensor":
        """Deserialize a PermutedTensor from IRPA"""
        from .tensors import REGISTERED_INFERENCE_TENSOR_CLASSES

        original_type_name = extra_properties["original_type"]
        permute_dims = tuple(extra_properties["permute_dims"])

        original_tensor_class = REGISTERED_INFERENCE_TENSOR_CLASSES[original_type_name]
        base_tensor = original_tensor_class.create_deserialized(
            name, raw_tensors, extra_properties
        )

        return cls(base_tensor, permute_dims, name)

    def add_to_archive(self, builder) -> InferenceTensorMetadata:
        """Add to archive by delegating to the base tensor"""
        return self._base_tensor.add_to_archive(builder)

    def is_deep_equal(self, other: Any, *, compare_name: bool = True) -> bool:
        """Deep equality check"""
        if not isinstance(other, PermutedTensor):
            return False
        if compare_name and self.name != other.name:
            return False
        return (
            self.permute_dims == other.permute_dims
            and self._base_tensor.is_deep_equal(
                other._base_tensor, compare_name=compare_name
            )
        )
