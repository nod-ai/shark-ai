// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_NODE_NODE_H
#define FUSILI_NODE_NODE_H

#include "fusili/context.h"
#include "fusili/logging.h"

#include <memory>

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
  // This is implicitly topologically sorted, as a result of
  // the functional API.
  std::vector<std::shared_ptr<INode>> subNodes_;

  virtual error_t preValidateNode() const { return {error_code_t::OK, ""}; }
  virtual error_t inferPropertiesNode() = 0;
  virtual error_t postValidateNode() const { return {error_code_t::OK, ""}; }

  error_t validateSubtree() {
    FUSILI_CHECK_ERROR(preValidateNode());
    FUSILI_CHECK_ERROR(inferPropertiesNode());
    for (const auto &subNode : subNodes_) {
      FUSILI_CHECK_ERROR(subNode->validateSubtree());
    }
    FUSILI_CHECK_ERROR(postValidateNode());
    return {error_code_t::OK, ""};
  }
};

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
