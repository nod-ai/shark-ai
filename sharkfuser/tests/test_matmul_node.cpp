// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace fusilli;

TEST_CASE("MatmulNode getName correctly propagates the attribute name",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;
  attr.setName("foo_matmul");

  MatmulNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_matmul");
}

TEST_CASE("MatmulNode getType returns correct type", "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;
  attr.setName("test_matmul");

  MatmulNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::Matmul);
}

TEST_CASE("MatmulNode preValidateNode detects missing attributes",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  SECTION("Input A missing") {
    MatmulNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Matmul input tensor A not set");
  }

  SECTION("Input B missing") {
    attr.setA(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    MatmulNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Matmul input tensor B not set");
  }

  SECTION("Output C missing") {
    attr.setA(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setB(std::make_shared<TensorAttr>(
        TensorAttr().setDim({3, 4}).setStride({4, 1})));
    MatmulNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Matmul output tensor C not set");
  }

  SECTION("All required attributes present") {
    attr.setA(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setB(std::make_shared<TensorAttr>(
        TensorAttr().setDim({3, 4}).setStride({4, 1})));
    attr.setC(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 4}).setStride({4, 1})));
    MatmulNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }
}

TEST_CASE("MatmulNode inferPropertiesNode when C is fully specified",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  int64_t m = 16, k = 32, n = 64;

  attr.setA(std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, k}).setStride({k, 1})));
  attr.setB(std::make_shared<TensorAttr>(
      TensorAttr().setDim({k, n}).setStride({n, 1})));
  attr.setC(std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, n}).setStride({n, 1})));

  MatmulNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
  FUSILLI_REQUIRE_OK(node.postValidateNode());

  auto cT = node.matmulAttr.getC();
  REQUIRE(cT->getDim() == std::vector<int64_t>{m, n});
  REQUIRE(cT->getStride() == std::vector<int64_t>{n, 1});
}

TEST_CASE("MatmulNode inferPropertiesNode when C is under-specified",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  int64_t m = 16, k = 32, n = 64;

  attr.setA(std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, k}).setStride({k, 1})));
  attr.setB(std::make_shared<TensorAttr>(
      TensorAttr().setDim({k, n}).setStride({n, 1})));
  // C is under-specified (dim/stride missing).
  attr.setC(std::make_shared<TensorAttr>());

  MatmulNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
  FUSILLI_REQUIRE_OK(node.postValidateNode());

  auto cT = node.matmulAttr.getC();
  REQUIRE(cT->getDim() == std::vector<int64_t>{m, n});
  REQUIRE(cT->getStride() == std::vector<int64_t>{n, 1});
}

TEST_CASE("MatmulNode inferPropertiesNode with batched matrices",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  int64_t batch = 8, m = 16, k = 32, n = 64;

  attr.setA(std::make_shared<TensorAttr>(
      TensorAttr().setDim({batch, m, k}).setStride({m * k, k, 1})));
  attr.setB(std::make_shared<TensorAttr>(
      TensorAttr().setDim({batch, k, n}).setStride({k * n, n, 1})));
  // C is under-specified (dim/stride missing).
  attr.setC(std::make_shared<TensorAttr>());

  MatmulNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
  FUSILLI_REQUIRE_OK(node.postValidateNode());

  auto cT = node.matmulAttr.getC();
  REQUIRE(cT->getDim() == std::vector<int64_t>{batch, m, n});
  REQUIRE(cT->getStride() == std::vector<int64_t>{m * n, n, 1});
}

TEST_CASE("MatmulNode inferPropertiesNode with broadcasted batch dimensions",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  int64_t m = 16, k = 32, n = 64;

  SECTION("A has batch dimension, B does not") {
    int64_t batch = 8;

    attr.setA(std::make_shared<TensorAttr>(
        TensorAttr().setDim({batch, m, k}).setStride({m * k, k, 1})));
    attr.setB(std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, n}).setStride({n, 1})));
    // C is under-specified (dim/stride missing).
    attr.setC(std::make_shared<TensorAttr>());

    MatmulNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());

    auto cT = node.matmulAttr.getC();
    // Output should have the broadcasted batch dimension
    REQUIRE(cT->getDim() == std::vector<int64_t>{batch, m, n});
    REQUIRE(cT->getStride() == std::vector<int64_t>{m * n, n, 1});
  }

  SECTION("B has batch dimension, A does not") {
    int64_t batch = 8;

    attr.setA(std::make_shared<TensorAttr>(
        TensorAttr().setDim({m, k}).setStride({k, 1})));
    attr.setB(std::make_shared<TensorAttr>(
        TensorAttr().setDim({batch, k, n}).setStride({k * n, n, 1})));
    // C is under-specified (dim/stride missing).
    attr.setC(std::make_shared<TensorAttr>());

    MatmulNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());

    auto cT = node.matmulAttr.getC();
    // Output should have the broadcasted batch dimension with contiguous stride
    REQUIRE(cT->getDim() == std::vector<int64_t>{batch, m, n});
    REQUIRE(cT->getStride() == std::vector<int64_t>{m * n, n, 1});
  }

  SECTION("A has batch dimension of 1, B has larger batch dimension") {
    int64_t batch = 8;

    attr.setA(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, m, k}).setStride({m * k, k, 1})));
    attr.setB(std::make_shared<TensorAttr>(
        TensorAttr().setDim({batch, k, n}).setStride({k * n, n, 1})));
    // C is under-specified (dim/stride missing).
    attr.setC(std::make_shared<TensorAttr>());

    MatmulNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());

    auto cT = node.matmulAttr.getC();
    // Output should have the broadcasted batch dimension (max of 1 and batch)
    REQUIRE(cT->getDim() == std::vector<int64_t>{batch, m, n});
    REQUIRE(cT->getStride() == std::vector<int64_t>{m * n, n, 1});
  }

  SECTION("Multiple batch dimensions with broadcasting") {
    int64_t batch1 = 4, batch2 = 8;

    attr.setA(std::make_shared<TensorAttr>(
        TensorAttr()
            .setDim({1, batch2, m, k})
            .setStride({batch2 * m * k, m * k, k, 1})));
    attr.setB(
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({batch1, 1, k, n})
                                         .setStride({k * n, k * n, n, 1})));
    // C is under-specified (dim/stride missing).
    attr.setC(std::make_shared<TensorAttr>());

    MatmulNode node(std::move(attr), ctx);
    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());

    auto cT = node.matmulAttr.getC();
    // Output should have broadcasted batch dimensions: max(1, batch1),
    // max(batch2, 1)
    REQUIRE(cT->getDim() == std::vector<int64_t>{batch1, batch2, m, n});
    REQUIRE(cT->getStride() ==
            std::vector<int64_t>{batch2 * m * n, m * n, n, 1});
  }
}

TEST_CASE("MatmulNode preValidate accepts arbitrary input strides",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  int64_t m = 16, k = 32, n = 64;

  auto aT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, k}).setStride({k, 1}).setName("A_contiguous"));

  auto bT = std::make_shared<TensorAttr>(
      TensorAttr()
          .setDim({k, n})
          .setStride(
              {k * n, k}) // Arbitrary layout (not contiguous or channels-last)
          .setName("B_arbitrary_layout"));

  attr.setA(aT).setB(bT).setC(std::make_shared<TensorAttr>());

  MatmulNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
  FUSILLI_REQUIRE_OK(node.postValidateNode());
}

TEST_CASE("MatmulNode postValidate accepts arbitrary output strides",
          "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  int64_t m = 16, k = 32, n = 64;

  auto aT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, k}).setStride({k, 1}).setName("A_contiguous"));

  auto bT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({k, n}).setStride({n, 1}).setName("B_contiguous"));

  auto cT = std::make_shared<TensorAttr>(
      TensorAttr()
          .setDim({m, n})
          .setStride(
              {m * n, m}) // Arbitrary layout (not contiguous or channels-last)
          .setName("C_arbitrary_layout"));

  attr.setA(aT).setB(bT).setC(cT);

  MatmulNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
  FUSILLI_REQUIRE_OK(node.postValidateNode());
}

TEST_CASE("MatmulNode rank checks", "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  SECTION("Input A must be at least rank 2") {
    auto aT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({16}).setStride({1}).setName("A_rank1"));

    auto bT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({16, 32}).setStride({32, 1}).setName("B_rank2"));

    auto cT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({32}).setStride({1}).setName("C_rank1"));

    attr.setA(aT).setB(bT).setC(cT);

    MatmulNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Matmul input tensor A must have a rank of at least 2");
  }

  SECTION("Input B must be at least rank 2") {
    auto aT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({16, 32}).setStride({32, 1}).setName("A_rank2"));

    auto bT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({32}).setStride({1}).setName("B_rank1"));

    auto cT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({16}).setStride({1}).setName("C_rank1"));

    attr.setA(aT).setB(bT).setC(cT);

    MatmulNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Matmul input tensor B must have a rank of at least 2");
  }

  SECTION("Output C must be at least rank 2") {
    auto aT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({16, 32}).setStride({32, 1}).setName("A_rank2"));

    auto bT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({32, 64}).setStride({64, 1}).setName("B_rank2"));

    auto cT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({16}).setStride({1}).setName("C_rank1"));

    attr.setA(aT).setB(bT).setC(cT);

    MatmulNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Matmul output tensor C must have a rank of at least 2");
  }
}

TEST_CASE("MatmulNode dimension compatibility checks", "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  SECTION("Inner dimensions must match") {
    int64_t m = 16, k1 = 32, k2 = 48, n = 64;

    auto aT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({m, k1}).setStride({k1, 1}).setName("A"));

    auto bT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k2, n}).setStride({n, 1}).setName("B"));

    auto cT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({m, n}).setStride({n, 1}).setName("C"));

    attr.setA(aT).setB(bT).setC(cT);

    MatmulNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Matmul input tensors A and B have incompatible inner dimensions "
            "(K): A has K=32, B has K=48");
  }

  SECTION("Valid dimensions") {
    int64_t m = 16, k = 32, n = 64;

    auto aT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({m, k}).setStride({k, 1}).setName("A"));

    auto bT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({k, n}).setStride({n, 1}).setName("B"));

    auto cT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({m, n}).setStride({n, 1}).setName("C"));

    attr.setA(aT).setB(bT).setC(cT);

    MatmulNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());
  }
}

TEST_CASE("MatmulNode postValidateNode dimension validation", "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  int64_t m = 16, k = 32, n = 64;

  auto aT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, k}).setStride({k, 1}).setName("A"));

  auto bT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({k, n}).setStride({n, 1}).setName("B"));

  // Wrong C dimensions - should be {m, n} but using {n, m}
  auto cT = std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, m}).setStride({m, 1}).setName("C"));

  attr.setA(aT).setB(bT).setC(cT);

  MatmulNode node(std::move(attr), ctx);

  FUSILLI_REQUIRE_OK(node.preValidateNode());
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto status = node.postValidateNode();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
  REQUIRE(status.getMessage() ==
          "Matmul output tensor C dimensions do not match the expected shapes "
          "inferred based on the input dimensions");
}

TEST_CASE("MatmulNode with different batch dimensions", "[matmul_node]") {
  Context ctx;
  MatmulAttr attr;

  SECTION("Both inputs batched with same batch size") {
    int64_t batch = 8, m = 16, k = 32, n = 64;

    auto aT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({batch, m, k}).setStride({m * k, k, 1}));
    auto bT = std::make_shared<TensorAttr>(
        TensorAttr().setDim({batch, k, n}).setStride({k * n, n, 1}));
    auto cT = std::make_shared<TensorAttr>();

    attr.setA(aT).setB(bT).setC(cT);

    MatmulNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());

    REQUIRE(node.matmulAttr.getC()->getDim() ==
            std::vector<int64_t>{batch, m, n});
  }

  SECTION("Multi-dimensional batch") {
    int64_t b1 = 2, b2 = 4, m = 16, k = 32, n = 64;

    auto aT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({b1, b2, m, k})
                                         .setStride({b2 * m * k, m * k, k, 1}));
    auto bT =
        std::make_shared<TensorAttr>(TensorAttr()
                                         .setDim({b1, b2, k, n})
                                         .setStride({b2 * k * n, k * n, n, 1}));
    auto cT = std::make_shared<TensorAttr>();

    attr.setA(aT).setB(bT).setC(cT);

    MatmulNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());

    REQUIRE(node.matmulAttr.getC()->getDim() ==
            std::vector<int64_t>{b1, b2, m, n});
  }
}
