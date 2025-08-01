// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the `Graph` class which derives from the
// `INode` class (like other nodes).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_H
#define FUSILLI_GRAPH_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/context.h"
#include "fusilli/logging.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/node/node.h"

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>

namespace fusilli {

class Graph : public INode {
public:
  Graph() : INode(Context{}) {}

  ErrorObject validate() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating Graph");

    // Validate nodes
    // This infers missing tensor properties such as dims,
    // stride, dtype based on context
    FUSILLI_CHECK_ERROR(validateSubtree());

    // Validate inputs
    // This has to happen after `validateSubtree` to infer any
    // missing properties on inputs first.
    for (const auto &input : fullGraphInputs_) {
      FUSILLI_CHECK_ERROR(input->validate());
    }

    // Validate outputs
    // This has to happen after `validateSubtree` to infer any
    // missing properties on outputs first.
    for (const auto &output : fullGraphOutputs_) {
      FUSILLI_CHECK_ERROR(output->validate());
    }

    FUSILLI_LOG_LABEL_ENDL("INFO: Graph validation completed successfully");
    isValidated_ = true;
    return ok();
  }

  ErrorOr<std::string> emitAsm() {
    FUSILLI_RETURN_ERROR_IF(
        !isValidated_, ErrorCode::NotValidated,
        "Graph must be validated before emitting MLIR assembly");
    FUSILLI_LOG_LABEL_ENDL("INFO: Emitting MLIR assembly for Graph");
    std::ostringstream oss;
    emitAsmSubtree(oss);
    FUSILLI_LOG_ENDL(oss.str());
    return oss.str();
  }

  std::string getName() const override final { return context.getName(); }
  Type getType() const override final { return Type::Composite; }

  Graph &setName(const std::string &name) {
    context.setName(name);
    return *this;
  }

  Graph &setIODataType(DataType type) {
    context.setIODataType(type);
    return *this;
  }

  Graph &setComputeDataType(DataType type) {
    context.setComputeDataType(type);
    return *this;
  }

  Graph &setIntermediateDataType(DataType type) {
    context.setIntermediateDataType(type);
    return *this;
  }

  // Declarations for tensor and op builder methods go here.
  // Definitions are towards the end of this file below.
  std::shared_ptr<TensorAttr> tensor(const TensorAttr &tensor);

  std::shared_ptr<TensorAttr> convFProp(const std::shared_ptr<TensorAttr> &x,
                                        const std::shared_ptr<TensorAttr> &w,
                                        ConvFPropAttr &attributes);

private:
  // This is set after `validate()` is run  at least once successfully.
  bool isValidated_ = false;

  // This is safe for post-insertion updates of TensorAttr (e.g. setting name
  // or other properties) since it uses the pointer value itself for hashing.
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphInputs_;
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphOutputs_;

  // These are sorted by the TensorAttr name, so post-insertion modification is
  // UB (undefined behavior). These are to be populated after the graph is fully
  // constructed and validated, and no further updates are expected.
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphInputsSorted_;
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphOutputsSorted_;

  std::shared_ptr<TensorAttr> outputTensor(const std::string &name) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Adding output tensor '"
                           << name << "' to Graph outputs");
    auto tensor = std::make_shared<TensorAttr>();
    tensor->setName(name).setIsVirtual(true);
    fullGraphOutputs_.insert(tensor);
    return tensor;
  }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating Graph");
    // Validate input/output names are unique (requirement for SSA).
    std::unordered_set<std::string> usedSymbols;
    for (const auto &t : fullGraphInputs_) {
      FUSILLI_RETURN_ERROR_IF(
          usedSymbols.find(t->getName()) != usedSymbols.end(),
          ErrorCode::InvalidAttribute,
          "Symbol name '" + t->getName() + "' already in use");
      usedSymbols.insert(t->getName());
    }
    for (const auto &t : fullGraphOutputs_) {
      FUSILLI_RETURN_ERROR_IF(
          usedSymbols.find(t->getName()) != usedSymbols.end(),
          ErrorCode::InvalidAttribute,
          "Symbol name '" + t->getName() + "' already in use");
      usedSymbols.insert(t->getName());
    }
    // Recursively validate node names are unique (requirement for SSA).
    FUSILLI_CHECK_ERROR(checkNodeNamesAreUnique(usedSymbols));

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for Graph");
    // Populate sorted inputs / outputs after graph is fully constructed
    // and pre-validated (to ensure no symbol conflict).
    fullGraphInputsSorted_.insert(fullGraphInputs_.begin(),
                                  fullGraphInputs_.end());
    fullGraphOutputsSorted_.insert(fullGraphOutputs_.begin(),
                                   fullGraphOutputs_.end());
    return ok();
  }

  ErrorObject postValidateNode() const override final { return ok(); }

  // MLIR assembly emitter helper methods
  std::string emitNodePreAsm() const override final;
  std::string emitNodePostAsm() const override final;
  std::string getOperandNamesAndTypesAsm() const override final;
  std::string getResultNamesAsm() const override final;
  std::string getResultTypesAsm() const override final;
};

// Given a TensorAttr, create a shared pointer and add it to the graph's
// inputs. This allows the graph to manage the lifetime of the input tensor.
inline std::shared_ptr<TensorAttr> Graph::tensor(const TensorAttr &tensor) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Adding input tensor '" << tensor.getName()
                                                       << "' to Graph inputs");
  auto tensorPtr = std::make_shared<TensorAttr>(tensor);
  fullGraphInputs_.insert(tensorPtr);
  return tensorPtr;
}

// Create a ConvFPropNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convFProp(const std::shared_ptr<TensorAttr> &x,
                 const std::shared_ptr<TensorAttr> &w,
                 ConvFPropAttr &convAttr) {
  // Populate names when not set
  if (convAttr.getName().empty())
    convAttr.setName("conv_fprop_" + std::to_string(subNodes_.size()));
  if (x->getName().empty())
    x->setName(convAttr.getName() + "_X");
  if (w->getName().empty())
    w->setName(convAttr.getName() + "_W");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvFPropNode '" << convAttr.getName()
                                                        << "' to Graph");

  // Set inputs
  convAttr.setX(x).setW(w);

  // Set outputs
  auto y = outputTensor(convAttr.getName() + "_Y");
  convAttr.setY(y);

  // Create node and add to Graph's subNodes_
  subNodes_.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(convAttr), context));

  return y;
}

} // namespace fusilli

#endif // FUSILLI_GRAPH_H
