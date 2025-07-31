// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the inline definitions for all the MLIR assembly
// generation methods on the `Graph`, `TensorAttr`, `INode` and derived node
// classes. It is meant to be a common place for all things ASM emitter related
// to make maintenance and future improvements easier.
//
// We use a combination of raw multi-line strings `R"(...)"` and `std::format`
// (from c++20) to implement a simple templating system for generating MLIR
// assembly code. This could be made better with a jinja2-like templating
// system but for now this gets us mostly what we need.
//
// Caution: An important foot-gun with `std::format` is to forget to double the
// brace for a literal `{` or `}`. i.e. always use `{{` for `{` and `}}` for `}`
// to disambiguate from the `{}` that `std::format` uses for replacements.
// If not you'll hit a compilation error like so:
//    "error: call to consteval function 'std::basic_format_string<char, ...'"
//    "is not a constant expression"
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_ASM_EMITTER_H
#define FUSILLI_ASM_EMITTER_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/types.h"

#include <cassert>
#include <format>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fusilli {

// An STL-style algorithm similar to std::for_each that applies a second
// functor between every pair of elements.
//
// This provides the control flow logic to, for example, print a
// comma-separated list:
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor each_fn, NullaryFunctor between_fn) {
  if (begin == end)
    return;
  each_fn(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    each_fn(*begin);
  }
}

// An overload of `interleave` which additionally accepts a SkipFunctor
// to skip certain elements based on a predicate.
//
// This provides the control flow logic to, for example, print a
// comma-separated list excluding "foo":
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; },
//              [&](std::string name) { return name == "foo"; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor, typename SkipFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor each_fn, NullaryFunctor between_fn,
                       SkipFunctor skip_fn) {
  if (begin == end)
    return;
  bool first = true;
  for (; begin != end; ++begin) {
    if (!skip_fn(*begin)) {
      if (!first)
        between_fn();
      first = false;
      each_fn(*begin);
    }
  }
}

// Map from Fusilli types to MLIR types.
static const std::unordered_map<DataType, std::string> DataTypeToMlirTypeAsm = {
    {DataType::Half, "f16"},       {DataType::BFloat16, "bf16"},
    {DataType::Float, "f32"},      {DataType::Double, "f64"},
    {DataType::Uint8, "ui8"},      {DataType::Int8, "si8"},
    {DataType::Int16, "si16"},     {DataType::Int32, "si32"},
    {DataType::Int64, "si64"},     {DataType::Boolean, "i1"},
    {DataType::FP8E5M2, "f8E5M2"},
};

// Given a vector of ints, returns the MLIR assembly for the
// `torch.constant.int` ops for each int value and the
// `torch.prim.ListConstruct` op wrapping these into a single
// value.
//
// For example if `getListOfIntOpsAsm` is called on these inputs:
//    listOfInts: {1, 2}
//    prefix: "stride"
//    suffix: "conv"
//
// It generates the following MLIR assembly:
//
//   %stride_val_0_conv = torch.constant.int 1
//   %stride_val_1_conv = torch.constant.int 2
//   %stride_conv = torch.prim.ListConstruct
//          %stride_val_0_conv, %stride_val_1_conv :
//              (!torch.int, !torch.int) -> !torch.list<int>
//
// The prefix is generally what attribute this refers to (e.g.
// padding, stride, dilation etc.) and the suffix is the node's
// unique name (for SSA disambiguation).
//
inline std::string getListOfIntOpsAsm(const std::vector<int64_t> &listOfInts,
                                      const std::string &prefix,
                                      const std::string &suffix) {
  std::ostringstream oss;
  std::vector<std::string> ssaValueNames;

  // Emit `torch.constant.int` ops for each int value
  for (size_t i = 0; i < listOfInts.size(); ++i) {
    std::string ssaValueName =
        "%" + prefix + "_val_" + std::to_string(i) + "_" + suffix;
    oss << ssaValueName << " = torch.constant.int " << listOfInts[i]
        << "\n    ";
    ssaValueNames.push_back(ssaValueName);
  }

  // Emit the ListConstruct op
  oss << "%" + prefix + "_" + suffix << " = torch.prim.ListConstruct ";
  // %val_0, %val_1, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn
      [&](std::string name) { oss << name; },
      // between_fn
      [&] { oss << ", "; });
  oss << " : (";
  // !torch.int, !torch.int, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn
      [&](std::string name) { oss << "!torch.int"; },
      // between_fn
      [&] { oss << ", "; });
  oss << ") -> !torch.list<int>\n";

  return oss.str();
}

//===----------------------------------------------------------------------===//
//
// TensorAttr ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits a ranked tensor type in MLIR assembly representation.
//
// This expects ranked tensors (non-scalar) as we blanket generate
// a `!torch.vtensor` type. The caller is responsible to check for
// this. In the future we may want to extend this (or add new methods)
// for scalar types (such as `!torch.int` or `!torch.bool`).
//
// Example:
//
//    TensorAttr t;
//    t.setName("tensor")
//      .setDataType(DataType::Float)
//      .setDim({2, 3})
//      .setStride({3, 1})
//
//    t.getValueTensorTypeAsm() generates "!torch.vtensor<[2,3],f32>"
//
inline std::string TensorAttr::getValueTensorTypeAsm() const {
  assert(!isScalar() &&
         "TensorAttr::getValueTensorTypeAsm expects a ranked tensor");
  assert(!getDim().empty() &&
         "TensorAttr::getValueTensorTypeAsm expects non-empty dims");
  assert(getDataType() != DataType::NotSet &&
         "TensorAttr::getValueTensorTypeAsm expects a valid data type");

  std::ostringstream oss;
  oss << "!torch.vtensor<[";
  const std::vector<int64_t> &dims = getDim();
  interleave(
      dims.begin(), dims.end(),
      // each_fn
      [&](int64_t dim) { oss << dim; },
      // between_fn
      [&] { oss << ","; });
  oss << "],";
  oss << DataTypeToMlirTypeAsm.at(getDataType());
  oss << ">";
  return oss.str();
}

// Emits an MLIR SSA value name starting with the `%` sigil based off the
// TensorAttr name but only using alphanumeric / underscore [A-Za-z0-9_]
// characters.
//
// `foo_Bar::X0` would become `%foo_BarX0`
//
inline std::string TensorAttr::getMlirSSAValueNameAsm() const {
  assert(!getName().empty() &&
         "TensorAttr name must not be empty for `getMlirSSAValueNameAsm`");

  std::string filtered = getName();
  std::erase_if(filtered,
                [](unsigned char c) { return !(std::isalnum(c) || c == '_'); });
  return "%" + filtered;
}

//===----------------------------------------------------------------------===//
//
// Graph ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits Graph's operand names and types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      func.func @main({}) -> ...
// with
//      "%arg0_image: !torch.vtensor<[16,128,64,64],f32>,
//       %arg1_filter: !torch.vtensor<[256,128,1,1],f32>"
//
// Order of operands is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphInputsSorted_`
// which sorts based on the name on the TensorAttrs.
//
inline std::string Graph::getOperandNamesAndTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphInputsSorted_.begin(), fullGraphInputsSorted_.end(),
      // each_fn
      [&](const std::shared_ptr<TensorAttr> &input) {
        oss << input->getMlirSSAValueNameAsm() << ": "
            << input->getValueTensorTypeAsm();
      },
      // between_fn
      [&] { oss << ", "; },
      // skip_fn
      [&](const std::shared_ptr<TensorAttr> &input) {
        // We only use the tensor inputs and not scalar (constants) as those
        // wouldn't be part of the main func.func signature but embedded as
        // constants in the IR.
        return input->isScalar();
      });
  return oss.str();
}

// Emits Graph's result names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      return {} : !torch.vtensor<[16,256,64,64],f32>
// with
//      "%result"
//
// Order of results is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphOutputsSorted_`
// which sorts based on the name on the TensorAttrs.
//
inline std::string Graph::getResultNamesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << output->getMlirSSAValueNameAsm();
      },
      // between_fn
      [&] { oss << ", "; },
      // skip_fn
      [&](const std::shared_ptr<TensorAttr> &output) {
        // We only want the final outputs in the return so ignore any virtual
        // tensors here as they're intermediates.
        return output->isVirtual();
      });
  return oss.str();
}

// Emits Graph's result types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      return %result : {}
// with
//      "!torch.vtensor<[16,256,64,64],f32>"
//
inline std::string Graph::getResultTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << output->getValueTensorTypeAsm();
      },
      // between_fn
      [&] { oss << ", "; },
      // skip_fn
      [&](const std::shared_ptr<TensorAttr> &output) {
        // We only want the final outputs in the return so ignore any virtual
        // tensors here as they're intermediates.
        return output->isVirtual();
      });
  return oss.str();
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string Graph::emitNodePreAsm() const {
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

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the post-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string Graph::emitNodePostAsm() const {
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

//===----------------------------------------------------------------------===//
//
// ConvFPropNode ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits ConvFPropNode's operand names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution {}, ...
// with
//      "%arg0_image, %arg1_filter"
//
inline std::string ConvFPropNode::getOperandNamesAsm() const {
  return convFPropAttr.getX()->getMlirSSAValueNameAsm() + ", " +
         convFPropAttr.getW()->getMlirSSAValueNameAsm();
}

// Emits ConvFPropNode's operand types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... : {}, ...
// with
//      "!torch.vtensor<[16,128,64,64],f32>, !torch.vtensor<[256,128,1,1],f32>"
//
inline std::string ConvFPropNode::getOperandTypesAsm() const {
  return convFPropAttr.getX()->getValueTensorTypeAsm() + ", " +
         convFPropAttr.getW()->getValueTensorTypeAsm();
}

// Emits ConvFPropNode's result names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      {} = torch.aten.convolution ...
// with
//      "%result"
//
inline std::string ConvFPropNode::getResultNamesAsm() const {
  return convFPropAttr.getY()->getMlirSSAValueNameAsm();
}

// Emits ConvFPropNode's result types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... -> {}
// with
//      "!torch.vtensor<[16,256,64,64],f32>"
//
inline std::string ConvFPropNode::getResultTypesAsm() const {
  return convFPropAttr.getY()->getValueTensorTypeAsm();
}

// Get strides in MLIR assembly format
inline std::string ConvFPropNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convFPropAttr.getName());
}

// Get padding in MLIR assembly format
inline std::string ConvFPropNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convFPropAttr.getName());
}

// Get dilation in MLIR assembly format
inline std::string ConvFPropNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convFPropAttr.getName());
}

// This gets called by the recursive `emitAsmSubtree()` method to emit
// the pre-assembly for each node (including the main Graph). The schema
// hard-codes things that are not customizable, and leaves the rest
// for template replacements using `std::format`. When modifying the
// schema, take extra caution about double bracing the curly brackets
// (refer to the comments at the top of this file for details).
inline std::string ConvFPropNode::emitNodePreAsm() const {
  // `torch.aten.convolution` signature from GeneratedTorchOps.td
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

  // Suffix the SSA names of internal values (constant attributes) using
  // the unique ConvFPropAttr name to avoid re-definition of names across
  // the overall MLIR assembly.
  std::string uniqueSSASuffix = convFPropAttr.getName();

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

} // namespace fusilli

#endif // FUSILLI_ASM_EMITTER_H
