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

#ifndef FUSILI_GRAPH_H
#define FUSILI_GRAPH_H

#include "fusili/attributes/tensor_attributes.h"
#include "fusili/context.h"
#include "fusili/logging.h"
#include "fusili/node/conv_node.h"
#include "fusili/node/node.h"

#include <memory>
#include <set>
#include <string>

namespace fusili {

class Graph : public INode {
public:
  Graph() : INode(Context{}) {}

  error_t validate() {
    FUSILI_LOG_LABEL_ENDL("INFO: Validating graph");

    // Validate inputs
    for (const auto &input : fullGraphInputs_) {
      FUSILI_CHECK_ERROR(input->validate());
    }

    // Validate nodes (this infers missing tensor properties)
    FUSILI_CHECK_ERROR(validateSubtree());

    // Validate outputs
    for (const auto &output : fullGraphOutputs_) {
      FUSILI_CHECK_ERROR(output->validate());
    }

    return {error_code_t::OK, ""};
  }

  std::string emitAsm() {
    FUSILI_LOG_LABEL_ENDL("INFO: Emitting MLIR assembly for graph");
    std::ostringstream oss;
    emitAsmSubtree(oss);
    FUSILI_LOG_ENDL(oss.str());
    return oss.str();
  }

  Type getType() override { return Type::Composite; }

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
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName> fullGraphInputs_;
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName> fullGraphOutputs_;

  std::shared_ptr<TensorAttr> outputTensor(const std::string &name) {
    auto tensor = std::make_shared<TensorAttr>();
    tensor->setName(name).setIsVirtual(true);
    fullGraphOutputs_.insert(tensor);
    return tensor;
  }

  error_t preValidateNode() const override final {
    return {error_code_t::OK, ""};
  }

  error_t inferPropertiesNode() override final {
    return {error_code_t::OK, ""};
  }

  error_t postValidateNode() const override final {
    return {error_code_t::OK, ""};
  }

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

} // namespace fusili

#endif // FUSILI_GRAPH_H
