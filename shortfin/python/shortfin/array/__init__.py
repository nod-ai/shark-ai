# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util

from _shortfin import lib as _sfl

# All dtype aliases.
opaque8 = _sfl.array.opaque8
opaque16 = _sfl.array.opaque16
opaque32 = _sfl.array.opaque32
opaque64 = _sfl.array.opaque64
bool8 = _sfl.array.bool8
int4 = _sfl.array.int4
sint4 = _sfl.array.sint4
uint4 = _sfl.array.uint4
int8 = _sfl.array.int8
sint8 = _sfl.array.sint8
uint8 = _sfl.array.uint8
int16 = _sfl.array.int16
sint16 = _sfl.array.sint16
uint16 = _sfl.array.uint16
int32 = _sfl.array.int32
sint32 = _sfl.array.sint32
uint32 = _sfl.array.uint32
int64 = _sfl.array.int64
sint64 = _sfl.array.sint64
uint64 = _sfl.array.uint64
float16 = _sfl.array.float16
float32 = _sfl.array.float32
float64 = _sfl.array.float64
bfloat16 = _sfl.array.bfloat16
complex64 = _sfl.array.complex64
complex128 = _sfl.array.complex128


base_array = _sfl.array.base_array
device_array = _sfl.array.device_array
read_barrier = _sfl.array.read_barrier
write_barrier = _sfl.array.write_barrier
disable_barrier = _sfl.array.disable_barrier
storage = _sfl.array.storage
DType = _sfl.array.DType

# Ops.
argmax = _sfl.array.argmax
argpartition = _sfl.array.argpartition
add = _sfl.array.add
ceil = _sfl.array.ceil
convert = _sfl.array.convert
divide = _sfl.array.divide
fill_randn = _sfl.array.fill_randn
floor = _sfl.array.floor
log_softmax = _sfl.array.log_softmax
multiply = _sfl.array.multiply
round = _sfl.array.round
subtract = _sfl.array.subtract
transpose = _sfl.array.transpose
trunc = _sfl.array.trunc
RandomGenerator = _sfl.array.RandomGenerator

__all__ = [
    # DType aliases.
    "opaque8",
    "opaque16",
    "opaque32",
    "opaque64",
    "bool8",
    "int4",
    "sint4",
    "uint4",
    "int8",
    "sint8",
    "uint8",
    "int16",
    "sint16",
    "uint16",
    "int32",
    "sint32",
    "uint32",
    "int64",
    "sint64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "complex64",
    "complex128",
    # Classes.
    "base_array",
    "device_array",
    "read_barrier",
    "write_barrier",
    "disable_barrier",
    "storage",
    "DType",
    # Ops.
    "add",
    "argmax",
    "argpartition",
    "ceil",
    "convert",
    "divide",
    "fill_randn",
    "floor",
    "log_softmax",
    "multiply",
    "round",
    "subtract",
    "transpose",
    "trunc",
    "RandomGenerator",
]

# Import nputils if numpy is present.
np_present = importlib.util.find_spec("numpy") is not None
if np_present:
    from . import _nputils as nputils

    __all__.append("nputils")
