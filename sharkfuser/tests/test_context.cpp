// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusili;

TEST_CASE("Context setters and getters", "[Context]") {
  Context ctx;

  SECTION("Default values") {
    REQUIRE(ctx.getComputeDataType() == DataType_t::NOT_SET);
    REQUIRE(ctx.getIntermediateDataType() == DataType_t::NOT_SET);
    REQUIRE(ctx.getIODataType() == DataType_t::NOT_SET);
    REQUIRE(ctx.getName() == "");
  }

  SECTION("Set and get compute_data_type") {
    ctx.setComputeDataType(DataType_t::FLOAT);
    REQUIRE(ctx.getComputeDataType() == DataType_t::FLOAT);
  }

  SECTION("Set and get intermediate_data_type") {
    ctx.setIntermediateDataType(DataType_t::DOUBLE);
    REQUIRE(ctx.getIntermediateDataType() == DataType_t::DOUBLE);
  }

  SECTION("Set and get io_data_type") {
    ctx.setIODataType(DataType_t::INT32);
    REQUIRE(ctx.getIODataType() == DataType_t::INT32);
  }

  SECTION("Set and get name") {
    ctx.setName("my_context");
    REQUIRE(ctx.getName() == "my_context");
  }

  SECTION("Method chaining") {
    auto &result = ctx.setComputeDataType(DataType_t::FLOAT)
                       .setIntermediateDataType(DataType_t::DOUBLE)
                       .setIODataType(DataType_t::INT64)
                       .setName("chain");
    REQUIRE(&result == &ctx); // Verify chaining returns same object
    REQUIRE(ctx.getComputeDataType() == DataType_t::FLOAT);
    REQUIRE(ctx.getIntermediateDataType() == DataType_t::DOUBLE);
    REQUIRE(ctx.getIODataType() == DataType_t::INT64);
    REQUIRE(ctx.getName() == "chain");
  }
}
