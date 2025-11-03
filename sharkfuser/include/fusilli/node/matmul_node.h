// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the matrix multiplication node
// `MatmulNode`.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_NODE_MATMUL_NODE_H
#define FUSILLI_NODE_MATMUL_NODE_H

#include "fusilli/attributes/matmul_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/node.h"
#include "fusilli/support/logging.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
// Helper functions for matrix multiplication nodes.
//===----------------------------------------------------------------------===//

// Infer the output shape of a matrix multiplication operation from the input
// shapes. For matrices A [..., M, K] and B [..., K, N], the output is [..., M,
// N].
inline std::vector<int64_t>
getMatmulInferredOutputShape(const std::vector<int64_t> &aDim,
                             const std::vector<int64_t> &bDim) {
  size_t aRank = aDim.size();
  size_t bRank = bDim.size();
  size_t outRank = std::max(aRank, bRank);

  std::vector<int64_t> cDim(outRank);

  // Handle batch dimensions (broadcast if necessary)
  size_t batchDims = outRank - 2;
  for (size_t i = 0; i < batchDims; ++i) {
    int64_t aDimVal = (i < aRank - 2) ? aDim[i] : 1;
    int64_t bDimVal = (i < bRank - 2) ? bDim[i] : 1;
    // Use the maximum of the two dimensions (broadcasting rule)
    cDim[i] = std::max(aDimVal, bDimVal);
  }

  // Matrix dimensions: M from A, N from B
  cDim[outRank - 2] = aDim[aRank - 2]; // M
  cDim[outRank - 1] = bDim[bRank - 1]; // N

  return cDim;
}

//===----------------------------------------------------------------------===//
// Matrix multiplication node.
//===----------------------------------------------------------------------===//

class MatmulNode : public NodeCRTP<MatmulNode> {
public:
  MatmulAttr matmulAttr;

  MatmulNode(MatmulAttr &&attr, const Context &ctx)
      : NodeCRTP(ctx), matmulAttr(std::move(attr)) {}

  const std::string &getName() const override final {
    return matmulAttr.getName();
  }
  Type getType() const override final { return Type::Matmul; }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating MatmulNode '"
                           << matmulAttr.getName() << "'");

    std::shared_ptr<TensorAttr> aT = matmulAttr.getA();
    std::shared_ptr<TensorAttr> bT = matmulAttr.getB();
    std::shared_ptr<TensorAttr> cT = matmulAttr.getC();

    // Ensure input and output tensors are set.
    FUSILLI_RETURN_ERROR_IF(!aT, ErrorCode::AttributeNotSet,
                            "Matmul input tensor A not set");
    FUSILLI_RETURN_ERROR_IF(!bT, ErrorCode::AttributeNotSet,
                            "Matmul input tensor B not set");
    FUSILLI_RETURN_ERROR_IF(!cT, ErrorCode::AttributeNotSet,
                            "Matmul output tensor C not set");

    size_t aRank = aT->getDim().size();
    size_t bRank = bT->getDim().size();

    // Rank checks on input tensors (must be at least rank 2).
    FUSILLI_RETURN_ERROR_IF(
        aRank < 2, ErrorCode::InvalidAttribute,
        "Matmul input tensor A must have a rank of at least 2");
    FUSILLI_RETURN_ERROR_IF(
        bRank < 2, ErrorCode::InvalidAttribute,
        "Matmul input tensor B must have a rank of at least 2");

    // Check that inner dimensions match (K dimension).
    const std::vector<int64_t> &aDim = aT->getDim();
    const std::vector<int64_t> &bDim = bT->getDim();
    int64_t aK = aDim[aRank - 1]; // Last dimension of A
    int64_t bK = bDim[bRank - 2]; // Second-to-last dimension of B

    FUSILLI_RETURN_ERROR_IF(
        aK != bK, ErrorCode::InvalidAttribute,
        "Matmul input tensors A and B have incompatible inner dimensions (K): "
        "A has K=" +
            std::to_string(aK) + ", B has K=" + std::to_string(bK));

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for MatmulNode '"
                           << matmulAttr.getName() << "'");

    matmulAttr.fillFromContext(context);

    std::shared_ptr<TensorAttr> aT = matmulAttr.getA();
    std::shared_ptr<TensorAttr> bT = matmulAttr.getB();
    std::shared_ptr<TensorAttr> cT = matmulAttr.getC();

    const std::vector<int64_t> &aDim = aT->getDim();
    const std::vector<int64_t> &bDim = bT->getDim();

    const std::vector<int64_t> &cDim = cT->getDim();
    const std::vector<int64_t> &cStride = cT->getStride();

    // Infer shape of output tensor.
    if (cDim.empty())
      cT->setDim(getMatmulInferredOutputShape(aDim, bDim));

    // Infer stride of output tensor.
    if (cStride.empty()) {
      cT->setStride(
          generateStrideFromDim(cDim, getContiguousStrideOrder(cDim.size())));
    }

    return ok();
  }

  ErrorObject postValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Post-Validating MatmulNode '"
                           << matmulAttr.getName() << "'");

    std::shared_ptr<TensorAttr> aT = matmulAttr.getA();
    std::shared_ptr<TensorAttr> bT = matmulAttr.getB();
    std::shared_ptr<TensorAttr> cT = matmulAttr.getC();

    size_t cRank = cT->getDim().size();

    // Rank checks
    FUSILLI_RETURN_ERROR_IF(
        cRank < 2, ErrorCode::InvalidAttribute,
        "Matmul output tensor C must have a rank of at least 2");

    FUSILLI_RETURN_ERROR_IF(
        cT->getDim() !=
            getMatmulInferredOutputShape(aT->getDim(), bT->getDim()),
        ErrorCode::InvalidAttribute,
        "Matmul output tensor C dimensions do not match the expected shapes "
        "inferred based on the input dimensions");

    // No layout restrictions - we support arbitrary strides through
    // permutations.

    return ok();
  }
};

} // namespace fusilli

#endif // FUSILLI_NODE_MATMUL_NODE_H
