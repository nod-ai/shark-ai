// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the element types used throughout Fusilli datastructures.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ATTRIBUTES_TYPES_H
#define FUSILLI_ATTRIBUTES_TYPES_H

#include "fusilli/external/torch_types.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace fusilli {

// Half precision floating point from Clang extensions.
// https://clang.llvm.org/docs/LanguageExtensions.html#half-precision-floating-point
// These should be supported by GCC as well.
// TODO(#2226): When on C++23, switch to using `std::float16_t`
// and `std::bfloat16_t` from <stdfloat> (C++23).
// https://en.cppreference.com/w/cpp/types/floating-point.html
using half = _Float16;
using bf16 = __bf16;

// Define a macro to iterate over all fusilli datatypes and the corresponding
// torch datatypes and mlir asm.
#define FUSILLI_FORALL_DATA_TYPES(_)                                           \
  _(Half, Half, "f16")                                                         \
  _(BFloat16, BFloat16, "bf16")                                                \
  _(Float, Float, "f32")                                                       \
  _(Double, Double, "f64")                                                     \
  _(Uint8, Byte, "ui8")                                                        \
  _(Int8, Char, "si8")                                                         \
  _(Int16, Short, "si16")                                                      \
  _(Int32, Int, "si32")                                                        \
  _(Int64, Long, "si64")                                                       \
  _(Boolean, Bool, "i1")                                                       \
  _(FP8E5M2, Float8_e5m2, "f8E5M2")

enum class DataType : uint8_t {
  NotSet,
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE) FUSILLI_TYPE,
  FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from Fusilli types to MLIR types.
static const std::unordered_map<DataType, std::string> kDataTypeToMlirTypeAsm =
    {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE)                       \
  {DataType::FUSILLI_TYPE, MLIR_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// Map from Fusilli types to Torch types.
static const std::unordered_map<DataType, torch_upstream::ScalarType>
    kDataTypeToTorchType = {
#define DEFINE_ENUM(FUSILLI_TYPE, TORCH_TYPE, MLIR_TYPE)                       \
  {DataType::FUSILLI_TYPE, torch_upstream::ScalarType::TORCH_TYPE},
        FUSILLI_FORALL_DATA_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_TYPES_H
