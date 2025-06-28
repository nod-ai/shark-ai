// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusili;

TEST_CASE("Tensor operations", "[tensor]") {
  Graph graph;
  graph.set_io_data_type(DataType_t::HALF)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  int64_t uid = 1;
  std::string name = "image";

  TensorAttr tensor_attr;
  REQUIRE(tensor_attr.validate().is_bad());

  // Placeholder for tensor tests
  REQUIRE(true);
}
