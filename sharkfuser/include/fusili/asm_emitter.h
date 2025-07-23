// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the inline definitions for all the MLIR assembly
// generation methods on the `Graph`, `TensorAttr` and `Node` (and derived)
// classes. It is meant to be a common place for all things ASM emitter related
// to make maintenance and future improvements easier.
//
//===----------------------------------------------------------------------===//

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

// Map from Fusili types to MLIR types.
static const std::unordered_map<DataType, std::string> DataTypeToMlirTypeAsm = {
    {DataType::Half, "f16"},       {DataType::BFloat16, "bf16"},
    {DataType::Float, "f32"},      {DataType::Double, "f64"},
    {DataType::Uint8, "ui8"},      {DataType::Int8, "si8"},
    {DataType::Int16, "si16"},     {DataType::Int32, "si32"},
    {DataType::Int64, "si64"},     {DataType::Boolean, "i1"},
    {DataType::FP8E5M2, "f8E5M2"},
};

// TensorAttr method that returns the MLIR assembly representation
// of the ranked tensor type for it.
//
// This expects ranked tensors (non-scalar) as we blanket generate
// a `!torch.vtensor` type. The caller is responsible to check for
// this. In the future we may want to extend this to scalar types
// (such as `!torch.int` or `!torch.bool`).
//
//  TensorAttr t;
//  t.setName("tensor")
//    .setDataType(DataType::Float)
//    .setDim({2, 3})
//    .setStride({3, 1})
//
//  t.getRankedTensorTypeAsm() == "!torch.vtensor<[2,3],f32>"
//
inline std::string TensorAttr::getRankedTensorTypeAsm() const {
  assert(!isScalar() &&
         "TensorAttr::getRankedTensorTypeAsm expects a ranked tensor");
  assert(!getDim().empty() &&
         "TensorAttr::getRankedTensorTypeAsm expects non-empty dims");
  assert(getDataType() != DataType::NotSet &&
         "TensorAttr::getRankedTensorTypeAsm expects a valid data type");

  std::ostringstream oss;
  oss << "!torch.vtensor<[";
  const std::vector<int64_t> &dims = getDim();
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0)
      oss << ",";
    oss << dims[i];
  }
  oss << "],";
  oss << DataTypeToMlirTypeAsm.at(getDataType());
  oss << ">";
  return oss.str();
}

// Converts a string to a MLIR SSA name starting with the `%` sigil
// and only containing alphanumeric / underscore [A-Za-z0-9_] characters
inline std::string TensorAttr::getMlirSSANameAsm() const {
  assert(!getName().empty() &&
         "TensorAttr name must not be empty for `getMlirSSANameAsm`");

  std::string filtered;
  for (char c : getName()) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      filtered += c;
  }
  return "%" + filtered;
}

inline std::string Graph::getOperandNamesAndTypesAsm() const {
  std::ostringstream oss;
  bool first = true;
  for (const auto &input : fullGraphInputs_) {
    if (!input->isScalar()) {
      if (!first)
        oss << ", ";
      first = false;
      oss << input->getMlirSSANameAsm() << ": "
          << input->getRankedTensorTypeAsm();
    }
  }
  return oss.str();
}

inline std::string Graph::getResultNamesAsm() const {
  std::ostringstream oss;
  bool first = true;
  for (const auto &output : fullGraphOutputs_) {
    if (!output->isVirtual()) {
      if (!first)
        oss << ", ";
      first = false;
      oss << output->getMlirSSANameAsm();
    }
  }
  return oss.str();
}

inline std::string Graph::getResultTypesAsm() const {
  std::ostringstream oss;
  bool first = true;
  for (const auto &output : fullGraphOutputs_) {
    if (!output->isVirtual()) {
      if (!first)
        oss << ", ";
      first = false;
      oss << output->getRankedTensorTypeAsm();
    }
  }
  return oss.str();
}

// We use a combination of raw multi-line strings `R"(...)"` and `std::format`
// (from c++20) to implement a simple templating system for generating mlir
// assembly code. This could be made better with a jinja2-like templating
// system but for now this gets us mostly what we need.

// Caution: An important foot-gun with `std::format` is to forget to double the
// brace for a literal `{` or `}`. i.e. always use `{{` for `{` and `}}` for `}`
// to disambiguate from the `{}` that `std::format` uses for replacements.
// If not you'll hit a compilation error like so:
//    "error: call to consteval function 'std::basic_format_string<char, ...'"
//    "is not a constant expression"

inline std::string Graph::emitNodeAsmPre() const {
  constexpr std::string_view schema = R"(
module @module {{
  func.func @main({0}) -> {1} attributes {{torch.assume_strict_symbolic_shapes}} {{
  )";

  std::string output = std::format(schema,
                                   getOperandNamesAndTypesAsm(), // {0}
                                   getResultTypesAsm()           // {1}
  );

  return output;
}

inline std::string Graph::emitNodeAsmPost() const {
  constexpr std::string_view schema = R"(
    return {0} : {1}
  }}
}}
  )";

  std::string output = std::format(schema,
                                   getResultNamesAsm(), // {0}
                                   getResultTypesAsm()  // {1}
  );

  return output;
}

inline std::string ConvFPropNode::getOperandNamesAsm() const {
  std::ostringstream oss;
  oss << attr.getX()->getMlirSSANameAsm();
  oss << ", ";
  oss << attr.getW()->getMlirSSANameAsm();
  return oss.str();
}

inline std::string ConvFPropNode::getOperandTypesAsm() const {
  std::ostringstream oss;
  oss << attr.getX()->getRankedTensorTypeAsm();
  oss << ", ";
  oss << attr.getW()->getRankedTensorTypeAsm();
  return oss.str();
}

inline std::string ConvFPropNode::getResultNamesAsm() const {
  std::ostringstream oss;
  oss << attr.getY()->getMlirSSANameAsm();
  return oss.str();
}

inline std::string ConvFPropNode::getResultTypesAsm() const {
  std::ostringstream oss;
  oss << attr.getY()->getRankedTensorTypeAsm();
  return oss.str();
}

inline std::string getListOfIntOpsAsm(const std::vector<int64_t> &listOfInts,
                                      const std::string &prefix,
                                      const std::string &suffix) {
  std::ostringstream oss;
  std::vector<std::string> ssaValueNames;

  // Emit `torch.constant.int` ops for each int value
  for (size_t i = 0; i < listOfInts.size(); ++i) {
    std::string ssa_name = prefix + "val_" + std::to_string(i) + "_" + suffix;
    oss << ssa_name << " = torch.constant.int " << listOfInts[i] << "\n    ";
    ssaValueNames.push_back(ssa_name);
  }

  // Emit the ListConstruct op
  oss << prefix + suffix << " = torch.prim.ListConstruct ";
  for (size_t i = 0; i < ssaValueNames.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << ssaValueNames[i];
  }
  oss << " : (";
  for (size_t i = 0; i < ssaValueNames.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << "!torch.int";
  }
  oss << ") -> !torch.list<int>\n";

  return oss.str();
}

inline std::string ConvFPropNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(attr.getStride(), /*prefix=*/"%stride_",
                            /*suffix=*/attr.getName());
}

inline std::string ConvFPropNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(attr.getPadding(), /*prefix=*/"%padding_",
                            /*suffix=*/attr.getName());
}

inline std::string ConvFPropNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(attr.getDilation(), /*prefix=*/"%dilation_",
                            /*suffix=*/attr.getName());
}

inline std::string ConvFPropNode::emitNodeAsmPre() const {
  // "torch.aten.convolution" signature from GeneratedTorchOps.td
  // https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td
  //
  //  def Torch_AtenConvolutionOp : Torch_Op<"aten.convolution", [
  //    ...
  //    let summary = "Generated op for `aten::convolution : (Tensor, Tensor,
  //    Tensor?, int[], int[], int[], bool, int[], int) -> (Tensor)`"; let
  //    arguments = (ins
  //      AnyTorchTensorType:$input,
  //      AnyTorchTensorType:$weight,
  //      AnyTorchOptionalTensorType:$bias,
  //      AnyTorchListOfTorchIntType:$stride,
  //      AnyTorchListOfTorchIntType:$padding,
  //      AnyTorchListOfTorchIntType:$dilation,
  //      Torch_BoolType:$transposed,
  //      AnyTorchListOfTorchIntType:$output_padding,
  //      Torch_IntType:$groups
  //    );
  //    let results = (outs
  //      AnyTorchOptionalTensorType:$result
  //    );
  //   ...
  constexpr std::string_view schema = R"(
    %bias_{0} = torch.constant.none
    {1}
    {2}
    {3}
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    %groups_{0} = torch.constant.int 1
    {4} = torch.aten.convolution {5}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0} : {6}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> {7}
    )";

  // Suffix the SSA names of internal values (constant attributes)
  // to avoid re-definition of value names across the whole assembly
  std::string uniqueSSASuffix = attr.getName();

  std::string output = std::format(schema,
                                   uniqueSSASuffix,      // {0}
                                   getStrideOpsAsm(),    // {1}
                                   getPaddingOpsAsm(),   // {2}
                                   getDilationOpsAsm(),  // {3}
                                   getResultNamesAsm(),  // {4}
                                   getOperandNamesAsm(), // {5}
                                   getOperandTypesAsm(), // {6}
                                   getResultTypesAsm()   // {7}
  );

  return output;
}

} // namespace fusili

#endif // FUSILI_ASM_EMITTER_H
