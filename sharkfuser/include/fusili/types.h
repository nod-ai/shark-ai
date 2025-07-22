// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_TYPES_H
#define FUSILI_TYPES_H

#include <string>
#include <unordered_map>

namespace fusili {

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

static const std::unordered_map<DataType, std::string> DataTypeToMlirType = {
    {DataType::Half, "f16"},       {DataType::BFloat16, "bf16"},
    {DataType::Float, "f32"},      {DataType::Double, "f64"},
    {DataType::Uint8, "ui8"},      {DataType::Int8, "si8"},
    {DataType::Int16, "si16"},     {DataType::Int32, "si32"},
    {DataType::Int64, "si64"},     {DataType::Boolean, "i1"},
    {DataType::FP8E5M2, "f8E5M2"},
};

} // namespace fusili

#endif // FUSILI_TYPES_H
