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
// (from C++20) to implement a simple templating system for generating MLIR
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

#ifndef FUSILLI_SUPPORT_ASM_EMITTER_H
#define FUSILLI_SUPPORT_ASM_EMITTER_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/attributes/types.h"
#include "fusilli/graph/graph.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/support/extras.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format> // C++20
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fusilli {

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
inline std::string getListOfIntOpsAsm(const std::vector<int64_t> &listOfInts,
                                      const std::string &prefix,
                                      const std::string &suffix) {
  std::ostringstream oss;
  std::vector<std::string> ssaValueNames;

  // Emit `torch.constant.int` ops for each int value.
  for (size_t i = 0; i < listOfInts.size(); ++i) {
    std::string ssaValueName =
        "%" + prefix + "_val_" + std::to_string(i) + "_" + suffix;
    oss << ssaValueName << " = torch.constant.int " << listOfInts[i]
        << "\n    ";
    ssaValueNames.push_back(ssaValueName);
  }

  // Emit the ListConstruct op.
  oss << "%" + prefix + "_" + suffix << " = torch.prim.ListConstruct ";
  // %val_0, %val_1, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn:
      [&](std::string name) { oss << name; },
      // between_fn:
      [&] { oss << ", "; });
  oss << " : (";
  // !torch.int, !torch.int, ...
  interleave(
      ssaValueNames.begin(), ssaValueNames.end(),
      // each_fn:
      [&](std::string name) { oss << "!torch.int"; },
      // between_fn:
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
// This expects ranked tensors (non-scalar) as we blanket generate a
// `!torch.vtensor` (or `!torch.tensor` if mutable) type. The caller
// is responsible to check for this. In the future we may want to extend
// this (or add new methods) for scalar types (such as `!torch.int` or
// `!torch.bool`).
//
// Example:
//
//    TensorAttr t;
//    t.setName("tensor")
//      .setDataType(DataType::Float)
//      .setDim({2, 3, 4})
//      .setStride({12, 1, 3})
//
//    t.getTensorTypeAsm(/*isValueTensor=*/true,
//                       /*useLogicalDims=*/true)
//        --> "!torch.vtensor<[2,3,4],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/false,
//                       /*useLogicalDims=*/true)
//        --> "!torch.tensor<[2,3,4],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/true,
//                       /*useLogicalDims=*/false)
//        --> "!torch.vtensor<[2,4,3],f32>"
//
//    t.getTensorTypeAsm(/*isValueTensor=*/false,
//                       /*useLogicalDims=*/false)
//        --> "!torch.tensor<[2,4,3],f32>"
inline std::string TensorAttr::getTensorTypeAsm(bool isValueTensor,
                                                bool useLogicalDims) const {
  assert(!isScalar() && "TensorAttr::getTensorTypeAsm expects a ranked tensor");
  assert(!getDim().empty() &&
         "TensorAttr::getTensorTypeAsm expects non-empty dims");
  assert(!getStride().empty() &&
         "TensorAttr::getTensorTypeAsm expects non-empty strides");
  assert(getDataType() != DataType::NotSet &&
         "TensorAttr::getTensorTypeAsm expects a valid data type");

  std::ostringstream oss;
  oss << (isValueTensor ? "!torch.vtensor<[" : "!torch.tensor<[");

  const std::vector<int64_t> &logicalDims = getDim();
  std::vector<int64_t> physicalDims(logicalDims.size());

  if (!useLogicalDims) {
    // Convert logical dims + stride into physical dims for MLIR assembly.
    //  dims [N, C, H, W] + strideOrder [3, 2, 1, 0] -> [N, C, H, W]
    //  dims [N, C, H, W] + strideOrder [3, 0, 2, 1] -> [N, H, W, C]
    std::vector<size_t> strideOrder(logicalDims.size());
    if (isContiguous())
      strideOrder = getContiguousStrideOrder(logicalDims.size());
    else if (isChannelsLast())
      strideOrder = getChannelsLastStrideOrder(logicalDims.size());
    else
      assert(false && "TensorAttr::getTensorTypeAsm unexpected stride order");
    for (size_t i = 0; i < logicalDims.size(); ++i)
      physicalDims[strideOrder[i]] = logicalDims[i];
    std::ranges::reverse(physicalDims); // C++20
  }

  // Emit dims in logical or physical order.
  interleave(
      useLogicalDims ? logicalDims.begin() : physicalDims.begin(),
      useLogicalDims ? logicalDims.end() : physicalDims.end(),
      // each_fn:
      [&](int64_t dim) { oss << dim; },
      // between_fn:
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
// `foo_Bar::X0` becomes `%foo_BarX0` if `isOutputAliased=false`.
// `foo_Bar::X0` becomes `%foo_BarX0_` if `isOutputAliased=true`.
inline std::string TensorAttr::getValueNameAsm(bool isOutputAliased) const {
  assert(!getName().empty() &&
         "TensorAttr name must not be empty for `getValueNameAsm`");

  std::string filtered = getName();
  std::erase_if(filtered, // C++20
                [](unsigned char c) { return !(std::isalnum(c) || c == '_'); });
  return "%" + filtered + (isOutputAliased ? "_" : "");
}

//===----------------------------------------------------------------------===//
//
// Graph ASM Emitter Methods
//
//===----------------------------------------------------------------------===//

// Emits Graph's operand names and types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      func.func @main(..., {}) -> ...
// with
//      "%arg0_image: !torch.vtensor<[16,128,64,64],f32>,
//       %arg1_filter: !torch.vtensor<[256,128,1,1],f32>"
//
// Order of operands is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphInputsSorted_`
// which sorts based on the name on the TensorAttrs.
inline std::string Graph::getOperandNamesAndTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphInputsSorted_.begin(), fullGraphInputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &input) {
        oss << input->getValueNameAsm() << ": " << input->getTensorTypeAsm();
      },
      // between_fn:
      [&] { oss << ", "; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &input) {
        // We only use the tensor inputs and not scalar (constants) as those
        // wouldn't be part of the main func.func signature but embedded as
        // constants in the IR.
        return input->isScalar();
      });
  return oss.str();
}

// Emits Graph's result names and types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      func.func @main({}, ...) -> ...
// with
//      "%result: !torch.tensor<[16,256,64,64],f32>
//
// Order of results is made to be deterministic, and it is
// determined by the sorting order used in `fullGraphOutputsSorted_`
// which sorts based on the name on the TensorAttrs.
inline std::string Graph::getResultNamesAndTypesAsm() const {
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << output->getValueNameAsm(/*isOutputAliased=*/true) << ": "
            << output->getTensorTypeAsm(/*isValueTensor=*/false);
      },
      // between_fn:
      [&] { oss << ", "; },
      // skip_fn:
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
  func.func @main({0}, {1}) attributes {{torch.assume_strict_symbolic_shapes}} {{
  )";

  std::string output = std::format(schema,
                                   getResultNamesAndTypesAsm(), // {0}
                                   getOperandNamesAndTypesAsm() // {1}
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
  std::ostringstream oss;
  interleave(
      fullGraphOutputsSorted_.begin(), fullGraphOutputsSorted_.end(),
      // each_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        oss << "torch.overwrite.tensor.contents "
            << output->getValueNameAsm(/*isOutputAliased=*/false)
            << " overwrites "
            << output->getValueNameAsm(/*isOutputAliased=*/true) << " : "
            << output->getTensorTypeAsm(/*isValueTensor=*/true) << ", "
            << output->getTensorTypeAsm(/*isValueTensor=*/false);
      },
      // between_fn:
      [&] { oss << "\n"; },
      // skip_fn:
      [&](const std::shared_ptr<TensorAttr> &output) {
        // We only want the final outputs in the return so ignore any virtual
        // tensors here as they're intermediates.
        return output->isVirtual();
      });

  constexpr std::string_view schema = R"(
    {0}

    return
  }}
}}
  )";

  std::string output = std::format(schema,
                                   oss.str() // {0}
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
inline std::string ConvFPropNode::getOperandNamesAsm() const {
  return convFPropAttr.getX()->getValueNameAsm() + "_perm" + ", " +
         convFPropAttr.getW()->getValueNameAsm() + "_perm";
}

// Emits ConvFPropNode's operand types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... : {}, ...
// with
//      "!torch.vtensor<[16,128,64,64],f32>, !torch.vtensor<[256,128,1,1],f32>"
inline std::string ConvFPropNode::getOperandTypesAsm() const {
  return convFPropAttr.getX()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true) +
         ", " +
         convFPropAttr.getW()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Emits ConvFPropNode's result names in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      {} = torch.aten.convolution ...
// with
//      "%result"
inline std::string ConvFPropNode::getResultNamesAsm() const {
  return convFPropAttr.getY()->getValueNameAsm();
}

// Emits ConvFPropNode's result types in MLIR assembly format.
//
// Its output is used to materialize the contents of {} in
//      %result = torch.aten.convolution ... -> {}
// with
//      "!torch.vtensor<[16,256,64,64],f32>"
inline std::string ConvFPropNode::getResultTypesAsm() const {
  return convFPropAttr.getY()->getTensorTypeAsm(/*isValueTensor=*/true,
                                                /*useLogicalDims=*/true);
}

// Get strides in MLIR assembly format.
inline std::string ConvFPropNode::getStrideOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getStride(), /*prefix=*/"stride",
                            /*suffix=*/convFPropAttr.getName());
}

// Get padding in MLIR assembly format.
inline std::string ConvFPropNode::getPaddingOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getPadding(), /*prefix=*/"padding",
                            /*suffix=*/convFPropAttr.getName());
}

// Get dilation in MLIR assembly format.
inline std::string ConvFPropNode::getDilationOpsAsm() const {
  return getListOfIntOpsAsm(convFPropAttr.getDilation(), /*prefix=*/"dilation",
                            /*suffix=*/convFPropAttr.getName());
}

// Get permute ops for input X in MLIR assembly format.
inline std::string ConvFPropNode::getPermuteXOpsAsm() const {
  std::ostringstream oss;

  std::string prefix = "permute_X";
  std::string suffix = convFPropAttr.getName();
  std::shared_ptr<TensorAttr> X = convFPropAttr.getX();

  // Emit permute dimensions based on layout.
  if (X->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(X->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(X->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  std::string output =
      std::format(schema,
                  X->getValueNameAsm(),        // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  X->getTensorTypeAsm(/*isValueTensor=*/true,
                                      /*useLogicalDims=*/false), // {2}
                  X->getTensorTypeAsm(/*isValueTensor=*/true,
                                      /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute ops for weight W in MLIR assembly format.
inline std::string ConvFPropNode::getPermuteWOpsAsm() const {
  std::ostringstream oss;

  std::string prefix = "permute_W";
  std::string suffix = convFPropAttr.getName();
  std::shared_ptr<TensorAttr> W = convFPropAttr.getW();

  // Emit permute dimensions based on layout.
  if (W->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(W->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getChannelsLastToContiguousPermuteOrder(W->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0}_perm = torch.aten.permute {0}, {1} : {2}, !torch.list<int> -> {3}
  )";

  std::string output =
      std::format(schema,
                  W->getValueNameAsm(),        // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  W->getTensorTypeAsm(/*isValueTensor=*/true,
                                      /*useLogicalDims=*/false), // {2}
                  W->getTensorTypeAsm(/*isValueTensor=*/true,
                                      /*useLogicalDims=*/true) // {3}
      );

  return oss.str() + output;
}

// Get permute ops for output Y in MLIR assembly format.
inline std::string ConvFPropNode::getPermuteYOpsAsm() const {
  std::ostringstream oss;

  std::string prefix = "permute_Y";
  std::string suffix = convFPropAttr.getName();
  std::shared_ptr<TensorAttr> Y = convFPropAttr.getY();

  // Emit permute dimensions based on layout.
  if (Y->isContiguous())
    oss << getListOfIntOpsAsm(
        getPreserveContiguousPermuteOrder(Y->getDim().size()), prefix, suffix);
  else
    oss << getListOfIntOpsAsm(
        getContiguousToChannelsLastPermuteOrder(Y->getDim().size()), prefix,
        suffix);

  // Emit the permute op itself.
  constexpr std::string_view schema = R"(
    {0} = torch.aten.permute {0}_perm, {1} : {2}, !torch.list<int> -> {3}
  )";

  std::string output =
      std::format(schema,
                  Y->getValueNameAsm(),        // {0}
                  "%" + prefix + "_" + suffix, // {1}
                  Y->getTensorTypeAsm(/*isValueTensor=*/true,
                                      /*useLogicalDims=*/true), // {2}
                  Y->getTensorTypeAsm(/*isValueTensor=*/true,
                                      /*useLogicalDims=*/false) // {3}
      );

  return oss.str() + output;
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
    %transposed_{0} = torch.constant.bool false
    %output_padding_{0} = torch.prim.ListConstruct  : () -> !torch.list<int>
    %groups_{0} = torch.constant.int 1
    {1}
    {2}
    {3}
    {4}
    {5}
    {6}_perm = torch.aten.convolution {7}, %bias_{0}, %stride_{0}, %padding_{0}, %dilation_{0}, %transposed_{0}, %output_padding_{0}, %groups_{0} : {8}, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> {9}
    {10}
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
                                   getPermuteXOpsAsm(),  // {4}
                                   getPermuteWOpsAsm(),  // {5}
                                   getResultNamesAsm(),  // {6}
                                   getOperandNamesAsm(), // {7}
                                   getOperandTypesAsm(), // {8}
                                   getResultTypesAsm(),  // {9}
                                   getPermuteYOpsAsm()   // {10}
  );

  return output;
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_ASM_EMITTER_H
