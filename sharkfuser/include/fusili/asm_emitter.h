// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_ASM_EMITTER_H
#define FUSILI_ASM_EMITTER_H

#include "fusili/attributes/tensor_attributes.h"
#include "fusili/graph.h"
#include "fusili/node/conv_node.h"
#include "fusili/types.h"

#include <cassert>
#include <format>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fusili {

// Given a TensorAttr, returns the assembly representation of the
// ranked tensor type for it.
//
// This expects ranked tensors (non-scalar) so the caller is
// responsible to check for this. This constraint exists because
// we generate a `!torch.vtensor` type. In the future it may be
// extended to generate scalar types (such as `!torch.int` or
// `!torch.bool`).
//
// Example:
//
//   TensorAttr t;
//   t.setName("tensor")
//      .setDataType(DataType::Float)
//      .setDim({2, 3})
//      .setStride({3, 1})
//
// getRankedTensorTypeAsm(t) would return "!torch.vtensor<[2,3],f32>"
//
inline std::string getRankedTensorTypeAsm(const TensorAttr &attr) {
  assert(!attr.isScalar() &&
         "TensorAttr must not be a scalar for `getRankedTensorTypeAsm`");
  assert(!attr.getDim().empty() &&
         "TensorAttr must have non-empty dims for `getRankedTensorTypeAsm`");
  assert(attr.getDataType() != DataType::NotSet &&
         "TensorAttr must have a valid data type for `getRankedTensorTypeAsm`");

  std::ostringstream oss;
  oss << "!torch.vtensor<[";
  const std::vector<int64_t> &dims = attr.getDim();
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0)
      oss << ",";
    oss << dims[i];
  }
  oss << "],";
  oss << DataTypeToMlirType.at(attr.getDataType());
  oss << ">";
  return oss.str();
}

// Converts a string to a MLIR SSA result name starting with the `%` sigil
// and only containing alphanumeric / underscore [A-Za-z0-9_] characters
inline std::string getMlirSSANameAsm(const std::string &name) {
  assert(!name.empty() && "Name must not be empty for `getMlirSSANameAsm`");

  std::string filtered;
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      filtered += c;
    }
  }
  return "%" + filtered;
}

inline std::string Graph::getOperandNamesAndTypesAsm() const {
  std::ostringstream oss;
  bool first = true;
  for (const auto &input : fullGraphInputs_) {
    if (!input->isScalar()) {
      if (!first) {
        oss << ", ";
      }
      first = false;
      oss << getMlirSSANameAsm(input->getName()) << ": "
          << getRankedTensorTypeAsm(*input);
    }
  }
  return oss.str();
}

inline std::string Graph::getResultTypesAsm() const {
  std::ostringstream oss;
  bool first = true;
  for (const auto &output : fullGraphOutputs_) {
    if (!output->isVirtual()) {
      if (!first) {
        oss << ", ";
      }
      first = false;
      oss << getRankedTensorTypeAsm(*output);
    }
  }
  return oss.str();
}

inline std::string Graph::getResultNamesAsm() const {
  std::ostringstream oss;
  bool first = true;
  for (const auto &output : fullGraphOutputs_) {
    if (!output->isVirtual()) {
      if (!first) {
        oss << ", ";
      }
      first = false;
      oss << getMlirSSANameAsm(output->getName());
    }
  }
  return oss.str();
}

// We use a combination of raw multi-line strings `R"(...)"` and `std::format`
// (from c++20) to implement a simple templating system for generating mlir
// assembly code. This could be made better with a jinja2-like templating
// system but for now this gets us mostly what we need.

// Caution: An important foot-gun here is to forget to double the brace for
// a literal `{` or `}`. i.e. always use `{{` for `{` and `}}` for `}` to
// disambiguate from the `{}` that `std::format` uses for replacements.
// If not you'll hit a compilation error like so:
//    "error: call to consteval function 'std::basic_format_string<char, ...'"
//    "is not a constant expression"

inline std::string Graph::emitNodeAsmPre() const {
  constexpr std::string_view schema = R"(
module @module {{
  func.func @main({0}) -> {1} attributes {{torch.assume_strict_symbolic_shapes}} {{
  )";

  std::string output = std::format(schema,
                                   // {0}
                                   getOperandNamesAndTypesAsm(),
                                   // {1}
                                   getResultTypesAsm());

  return output;
}

inline std::string Graph::emitNodeAsmPost() const {
  constexpr std::string_view schema = R"(
    return {0} : {1}
  }}
}}
  )";

  std::string output = std::format(schema,
                                   // {0}
                                   getResultNamesAsm(),
                                   // {1}
                                   getResultTypesAsm());

  return output;
}

inline std::string ConvFPropNode::emitNodeAsmPre() const {
  return R"(
    %false = torch.constant.bool false
    %int0 = torch.constant.int 0
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %4 = torch.aten.convolution %arg0, %arg1, %none, %0, %1, %2, %false, %3, %int1 : !torch.vtensor<[16,128,64,64],f32>, !torch.vtensor<[256,128,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[16,256,64,64],f32>
    )";
}

inline std::string ConvFPropNode::emitNodeAsmPost() const { return ""; }

} // namespace fusili

#endif // FUSILI_ASM_EMITTER_H
