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

enum class DataType {
  NotSet,
  Half,
  BFloat16,
  Float,
  Double,
  Uint8,
  Int8,
  Int16,
  Int32,
  Int64,
  Boolean,
  FP8E5M2,
};

// Map from Fusilli types to MLIR types.
static const std::unordered_map<DataType, std::string> DataTypeToMlirTypeAsm = {
    {DataType::Half, "f16"},       {DataType::BFloat16, "bf16"},
    {DataType::Float, "f32"},      {DataType::Double, "f64"},
    {DataType::Uint8, "ui8"},      {DataType::Int8, "si8"},
    {DataType::Int16, "si16"},     {DataType::Int32, "si32"},
    {DataType::Int64, "si64"},     {DataType::Boolean, "i1"},
    {DataType::FP8E5M2, "f8E5M2"},
};

} // namespace fusilli

#endif // FUSILLI_ATTRIBUTES_TYPES_H
