// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusilli;

TEST_CASE("Context setters and getters", "[Context]") {
  Context ctx;

  SECTION("Default values") {
    REQUIRE(ctx.getComputeDataType() == DataType::NotSet);
    REQUIRE(ctx.getIntermediateDataType() == DataType::NotSet);
    REQUIRE(ctx.getIODataType() == DataType::NotSet);
    REQUIRE(ctx.getName() == "");
  }

  SECTION("Set and get compute_data_type") {
    ctx.setComputeDataType(DataType::Float);
    REQUIRE(ctx.getComputeDataType() == DataType::Float);
  }

  SECTION("Set and get intermediate_data_type") {
    ctx.setIntermediateDataType(DataType::Double);
    REQUIRE(ctx.getIntermediateDataType() == DataType::Double);
  }

  SECTION("Set and get io_data_type") {
    ctx.setIODataType(DataType::Int32);
    REQUIRE(ctx.getIODataType() == DataType::Int32);
  }

  SECTION("Set and get name") {
    ctx.setName("my_context");
    REQUIRE(ctx.getName() == "my_context");
  }

  SECTION("Method chaining") {
    auto &result = ctx.setComputeDataType(DataType::Float)
                       .setIntermediateDataType(DataType::Double)
                       .setIODataType(DataType::Int64)
                       .setName("chain");
    REQUIRE(&result == &ctx); // Verify chaining returns same object
    REQUIRE(ctx.getComputeDataType() == DataType::Float);
    REQUIRE(ctx.getIntermediateDataType() == DataType::Double);
    REQUIRE(ctx.getIODataType() == DataType::Int64);
    REQUIRE(ctx.getName() == "chain");
  }
}
