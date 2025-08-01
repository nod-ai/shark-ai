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

#ifndef FUSILLI_NODE_NODE_H
#define FUSILLI_NODE_NODE_H

#include "fusilli/context.h"
#include "fusilli/logging.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>

namespace fusilli {

class INode {
public:
  enum class Type {
    Composite,
    Convolution,
  };

  explicit INode(const Context &ctx) : context(ctx) {}
  virtual ~INode() = default;

  virtual std::string getName() const = 0;
  virtual Type getType() const = 0;

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
  virtual ErrorObject preValidateNode() const { return ok(); }
  virtual ErrorObject inferPropertiesNode() = 0;
  virtual ErrorObject postValidateNode() const { return ok(); }

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
  ErrorObject validateSubtree() {
    FUSILLI_CHECK_ERROR(preValidateNode());
    FUSILLI_CHECK_ERROR(inferPropertiesNode());
    for (const auto &subNode : subNodes_) {
      FUSILLI_CHECK_ERROR(subNode->validateSubtree());
    }
    FUSILLI_CHECK_ERROR(postValidateNode());
    return ok();
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

  // Recursively check that names of nodes and their sub nodes
  // are unique to avoid re-definition of SSA values during
  // MLIR ASM generation.
  ErrorObject
  checkNodeNamesAreUnique(std::unordered_set<std::string> &usedSymbols) const {
    for (const auto &subNode : subNodes_) {
      FUSILLI_RETURN_ERROR_IF(
          usedSymbols.find(subNode->getName()) != usedSymbols.end(),
          ErrorCode::InvalidAttribute,
          "Symbol name '" + subNode->getName() + "' already in use");
      usedSymbols.insert(subNode->getName());
      FUSILLI_CHECK_ERROR(subNode->checkNodeNamesAreUnique(usedSymbols));
    }
    return ok();
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

} // namespace fusilli

#endif // FUSILLI_NODE_NODE_H
