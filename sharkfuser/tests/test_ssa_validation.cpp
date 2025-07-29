// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusili;

TEST_CASE("Multiple inputs use same name", "[graph][ssa]") {
  Graph graph;
  graph.setIODataType(DataType::Half);

  auto X =
      graph.tensor(TensorAttr().setName("image").setDim({1}).setStride({1}));

  auto W = graph.tensor(
      TensorAttr().setName("image").setDim({2, 2}).setStride({2, 1}));

  auto status = graph.validate();
  REQUIRE(status.isFailure());
  REQUIRE(status.getCode() == error_code_t::InvalidAttribute);
  REQUIRE(status.getMessage() == "Tensor with name 'image' already exists");
}
