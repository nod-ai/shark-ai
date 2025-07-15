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
  graph.set_io_data_type(DataType_t::HALF)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  int64_t uid = 1;
  std::string name = "image";

  auto X = graph.tensor(TensorAttr()
                            .set_name(name)
                            .set_dim({8, 32, 16, 16})
                            .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                            .set_uid(uid));

  // A new TensorAttr to populate via querying the graph
  TensorAttr t;
  REQUIRE(t.get_name() == "");
  REQUIRE(graph.query_tensor_of_uid(uid, t).is_ok());
  REQUIRE(t.get_name() == X->get_name());
}
