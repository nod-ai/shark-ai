# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Union
import os

import numpy as np
import torch

from gguf import GGUFReader, GGUFValueType, ReaderField

from iree.turbine.aot import (
    ExternalTensorTrait,
)

from amdsharktank.utils.logging import get_logger

from ..tensors import (
    DefaultPrimitiveTensor,
    InferenceTensor,
)

from ..theta import (
    Dataset,
    Theta,
)

from . import layouts

__all__ = [
    "load_file",
    "load_properties",
]

logger = get_logger("amdsharktank.gguf")


def _sanitize_scalar(scalar):
    if isinstance(scalar, np.generic):
        return scalar.tolist()  # Returns Python scalar type, not a list.
    return scalar


def _load_array(field: ReaderField) -> list:
    if len(field.types) != 2:
        raise ValueError(f"Unsupported array type {field.types}")
    element_type = field.types[1]
    if element_type == GGUFValueType.STRING:
        return [
            str(bytes(field.parts[parts_index]), encoding="utf8")
            for parts_index in field.data
        ]
    elif element_type in GGUFReader.gguf_scalar_to_np:
        return [
            _sanitize_scalar(field.parts[parts_index][0]) for parts_index in field.data
        ]
    else:
        raise ValueError(f"Unsupported array element type f{element_type}")


def _load_properties(reader: GGUFReader) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema": "GGUF",
    }

    # Extract hyper-parameters. Adapted from gguf-dump.py
    for field in reader.fields.values():
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                properties[field.name] = str(bytes(field.parts[-1]), encoding="utf8")
            elif field.types[0] in reader.gguf_scalar_to_np:
                properties[field.name] = _sanitize_scalar(field.parts[-1][0])
        elif field.types[0] == GGUFValueType.ARRAY:
            properties[field.name] = _load_array(field)
        else:
            raise ValueError(f"Invalid field type.")
    return properties


_quantized_types = {
    "Q4_1": layouts.Q4_1,
    "Q4_K": layouts.Q4_K,
    "Q5_K": layouts.Q5_K,
    "Q6_K": layouts.Q6_K,
    "Q8_0": layouts.Q8_0,
}


def _externalize_tensor(
    name: str, data: np.memmap, logical_shape: Optional[list[int]] = None
) -> torch.Tensor:
    # Important: The annotation tag must be set on the actual leaf tensor
    # which is stored in the root theta. This means that any shaping or
    # data type massaging has to happen *before* annotating.
    if logical_shape is not None:
        data_tensor = torch.as_tensor(data.reshape(logical_shape))
    else:
        data_tensor = torch.as_tensor(data)
    ExternalTensorTrait(external_name=name, external_scope="").set(data_tensor)
    return data_tensor


def _wrap_tensor(
    name: str, logical_shape: list[int], type_name: str, data: np.memmap
) -> InferenceTensor:
    # Gguf internally optimizes for constant RHS and stores all weights
    # transposed. So we reverse the reported logical shape. Most operations
    # are then logically done with a transposed RHS.
    # TODO: This needs some more investigation to ensure that it is in fact
    # always true.
    logical_shape = list(reversed(logical_shape))
    if type_name in ["F16", "F32", "F64"]:
        return DefaultPrimitiveTensor(
            name=name, data=_externalize_tensor(name, data, logical_shape)
        )

    if type_name == "BF16":
        assert data.dtype == np.uint8
        return DefaultPrimitiveTensor(
            name=name,
            data=_externalize_tensor(name, data.view(np.int16), logical_shape).view(
                dtype=torch.bfloat16
            ),
        )

    quantized_type = _quantized_types.get(type_name)
    if quantized_type is not None:
        return quantized_type(
            name=name,
            raw=_externalize_tensor(name, data),
            shape=logical_shape,
        )

    raise ValueError(f"Unsupported gguf tensor type: {type_name}")


def load_file(gguf_path: Union[str, os.PathLike]) -> Dataset:
    reader = GGUFReader(gguf_path)
    logger.info(
        "Loading gguf file %s (%d fields, %d tensors)",
        gguf_path,
        len(reader.fields),
        len(reader.tensors),
    )
    properties = _load_properties(reader)

    # Extract tensors.
    tensors: dict[str, InferenceTensor] = {}
    for tensor in reader.tensors:
        shape = [int(d) for d in tensor.shape]
        gguf_tensor = _wrap_tensor(
            name=tensor.name,
            logical_shape=list(shape),
            type_name=tensor.tensor_type.name,
            data=tensor.data,  # type: ignore
        )
        tensors[tensor.name] = gguf_tensor
    root_theta = Theta(tensors)
    return Dataset(properties=properties, root_theta=root_theta)


def load_properties(gguf_path: Union[str, os.PathLike], /) -> dict[str, Any]:
    reader = GGUFReader(gguf_path)
    return _load_properties(reader)
