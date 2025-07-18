// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace fusili;

TEST_CASE("Tensor query based on uid", "[tensor]") {
  Graph graph;
  graph.setIODataType(DataType::Half)
      .setIntermediateDataType(DataType::Float)
      .setComputeDataType(DataType::Float);

  int64_t uid = 1;
  std::string name = "image";

  auto X = graph.tensor(TensorAttr()
                            .setName(name)
                            .setDim({8, 32, 16, 16})
                            .setStride({32 * 16 * 16, 1, 32 * 16, 32})
                            .setUid(uid));

  // A new TensorAttr to populate via querying the graph
  TensorAttr t;
  REQUIRE(t.getName() == "");
  REQUIRE(graph.queryTensorOfUid(uid, t).isOk());
  REQUIRE(t.getName() == X->getName());
}
