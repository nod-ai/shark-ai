# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import struct
import numpy as np

# map numpy dtype -> (iree dtype, struct.pack format str)
dtype_map = {
    np.dtype("int64"): ("si64", "q"),
    np.dtype("uint64"): ("ui64", "Q"),
    np.dtype("int32"): ("si32", "i"),
    np.dtype("uint32"): ("ui32", "I"),
    np.dtype("int16"): ("si16", "h"),
    np.dtype("uint16"): ("ui16", "H"),
    np.dtype("int8"): ("si8", "b"),
    np.dtype("uint8"): ("ui8", "B"),
    np.dtype("float64"): ("f64", "d"),
    np.dtype("float32"): ("f32", "f"),
    np.dtype("float16"): ("f16", "e"),
    np.dtype("bool"): ("i1", "?"),
}


def pack_np_ndarray(ndarray: np.ndarray):
    mylist = ndarray.flatten().tolist()
    dtype = ndarray.dtype
    assert dtype in dtype_map
    return struct.pack(f"{len(mylist)}{dtype_map[dtype][1]}", *mylist)


def write_ndarray_to_bin(ndarray: np.ndarray, file: Path):
    with open(file, "wb") as f:
        packed_ndarray = pack_np_ndarray(ndarray)
        f.write(packed_ndarray)
