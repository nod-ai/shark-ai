// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("MatmulAttr default constructor", "[matmul_attr]") {
  MatmulAttr attr;
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("MatmulAttr setters and getters", "[matmul_attr]") {
  MatmulAttr attr;

  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());

  auto a = std::make_shared<TensorAttr>(1.0f);
  auto b = std::make_shared<TensorAttr>(2.0f);
  auto c = std::make_shared<TensorAttr>(3.0f);

  attr.setA(a).setB(b).setC(c);

  REQUIRE(attr.inputs.size() == 2);
  REQUIRE(attr.outputs.size() == 1);

  REQUIRE(attr.getA() == a);
  REQUIRE(attr.getB() == b);
  REQUIRE(attr.getC() == c);

  REQUIRE(attr.getA()->getDataType() == DataType::Float);
  REQUIRE(attr.getB()->getDataType() == DataType::Float);
  REQUIRE(attr.getC()->getDataType() == DataType::Float);

  REQUIRE(attr.getA()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getB()->getDim() == std::vector<int64_t>{1});
  REQUIRE(attr.getC()->getDim() == std::vector<int64_t>{1});

  REQUIRE(attr.getA()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getB()->getStride() == std::vector<int64_t>{1});
  REQUIRE(attr.getC()->getStride() == std::vector<int64_t>{1});

  REQUIRE(attr.getA()->isScalar() == true);
  REQUIRE(attr.getB()->isScalar() == true);
  REQUIRE(attr.getC()->isScalar() == true);

  REQUIRE(attr.getA()->isVirtual() == false);
  REQUIRE(attr.getB()->isVirtual() == false);
  REQUIRE(attr.getC()->isVirtual() == false);
}

TEST_CASE("MatmulAttr with matrix tensors", "[matmul_attr]") {
  MatmulAttr attr;

  int64_t m = 4, k = 8, n = 16;

  auto a = std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, k}).setStride({k, 1}).setName("A"));
  auto b = std::make_shared<TensorAttr>(
      TensorAttr().setDim({k, n}).setStride({n, 1}).setName("B"));
  auto c = std::make_shared<TensorAttr>(
      TensorAttr().setDim({m, n}).setStride({n, 1}).setName("C"));

  attr.setA(a).setB(b).setC(c).setName("matmul_test");

  REQUIRE(attr.getName() == "matmul_test");
  REQUIRE(attr.getA()->getDim() == std::vector<int64_t>{m, k});
  REQUIRE(attr.getB()->getDim() == std::vector<int64_t>{k, n});
  REQUIRE(attr.getC()->getDim() == std::vector<int64_t>{m, n});
}

TEST_CASE("MatmulAttr with batched tensors", "[matmul_attr]") {
  MatmulAttr attr;

  int64_t batch = 32, m = 64, k = 128, n = 256;

  auto a = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({batch, m, k})
                                            .setStride({m * k, k, 1})
                                            .setName("A_batched"));
  auto b = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({batch, k, n})
                                            .setStride({k * n, n, 1})
                                            .setName("B_batched"));
  auto c = std::make_shared<TensorAttr>(TensorAttr()
                                            .setDim({batch, m, n})
                                            .setStride({m * n, n, 1})
                                            .setName("C_batched"));

  attr.setA(a).setB(b).setC(c).setName("batched_matmul");

  REQUIRE(attr.getName() == "batched_matmul");
  REQUIRE(attr.getA()->getDim() == std::vector<int64_t>{batch, m, k});
  REQUIRE(attr.getB()->getDim() == std::vector<int64_t>{batch, k, n});
  REQUIRE(attr.getC()->getDim() == std::vector<int64_t>{batch, m, n});
}
