// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the `INode` and `NodeCRTP` classes which
// serve as the interfaces for individual op nodes as well as the main graph.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILI_NODE_NODE_H
#define FUSILI_NODE_NODE_H

#include "fusili/context.h"
#include "fusili/logging.h"

#include <memory>
#include <sstream>
#include <string>

namespace fusili {

class INode {
public:
  enum class Type {
    Composite,
    Convolution,
  };

  explicit INode(const Context &ctx) : context(ctx) {}
  virtual ~INode() = default;

  virtual Type getType() = 0;

  Context context;

protected:
  Type tag_;

  // This is a list of sub-nodes that this node may contain.
  // It is implicitly topologically sorted, as a result of
  // the functional API.
  std::vector<std::shared_ptr<INode>> subNodes_;

  // Virtual functions to be overridden by derived classes.
  // `inferPropertiesNode` is a pure virtual function and has
  // to be overridden.
  virtual error_t preValidateNode() const { return {error_code_t::OK, ""}; }
  virtual error_t inferPropertiesNode() = 0;
  virtual error_t postValidateNode() const { return {error_code_t::OK, ""}; }

  // MLIR assembly emitter helper methods to be provided
  // by each node as needed
  virtual std::string emitNodePreAsm() const { return ""; };
  virtual std::string emitNodePostAsm() const { return ""; };
  virtual std::string getOperandNamesAsm() const { return ""; };
  virtual std::string getOperandTypesAsm() const { return ""; };
  virtual std::string getOperandNamesAndTypesAsm() const { return ""; };
  virtual std::string getResultNamesAsm() const { return ""; };
  virtual std::string getResultTypesAsm() const { return ""; };

  // Recursively validate the node and its sub nodes
  error_t validateSubtree() {
    FUSILI_CHECK_ERROR(preValidateNode());
    FUSILI_CHECK_ERROR(inferPropertiesNode());
    for (const auto &subNode : subNodes_) {
      FUSILI_CHECK_ERROR(subNode->validateSubtree());
    }
    FUSILI_CHECK_ERROR(postValidateNode());
    return {error_code_t::OK, ""};
  }

  // Recursively emit MLIR assembly for the node and its sub nodes
  // allowing for composite ops to expand into their own regions
  // containing sub ops.
  void emitAsmSubtree(std::ostringstream &oss) {
    oss << emitNodePreAsm();
    for (const auto &subNode : subNodes_) {
      subNode->emitAsmSubtree(oss);
    }
    oss << emitNodePostAsm();
  }
};

// It uses the CRTP pattern (aka F-bound polymorphism):
// https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
template <typename DerivedT> class NodeCRTP : public INode {
protected:
  // Allow derived NodeCRTP classes to use the INode constructor
  using INode::INode;

private:
  DerivedT &self() { return static_cast<DerivedT &>(*this); }
  const DerivedT &self() const { return static_cast<const DerivedT &>(*this); }
};

} // namespace fusili

#endif // FUSILI_NODE_NODE_H
