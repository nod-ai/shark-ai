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
    REQUIRE(ctx.get_compute_data_type() == DataType_t::NOT_SET);
    REQUIRE(ctx.get_intermediate_data_type() == DataType_t::NOT_SET);
    REQUIRE(ctx.get_io_data_type() == DataType_t::NOT_SET);
    REQUIRE(ctx.get_name() == "");
  }

  SECTION("Set and get compute_data_type") {
    ctx.set_compute_data_type(DataType_t::FLOAT);
    REQUIRE(ctx.get_compute_data_type() == DataType_t::FLOAT);
  }

  SECTION("Set and get intermediate_data_type") {
    ctx.set_intermediate_data_type(DataType_t::DOUBLE);
    REQUIRE(ctx.get_intermediate_data_type() == DataType_t::DOUBLE);
  }

  SECTION("Set and get io_data_type") {
    ctx.set_io_data_type(DataType_t::INT32);
    REQUIRE(ctx.get_io_data_type() == DataType_t::INT32);
  }

  SECTION("Set and get name") {
    ctx.set_name("my_context");
    REQUIRE(ctx.get_name() == "my_context");
  }

  SECTION("Method chaining") {
    auto &result = ctx.set_compute_data_type(DataType_t::FLOAT)
                       .set_intermediate_data_type(DataType_t::DOUBLE)
                       .set_io_data_type(DataType_t::INT64)
                       .set_name("chain");
    REQUIRE(&result == &ctx); // Verify chaining returns same object
    REQUIRE(ctx.get_compute_data_type() == DataType_t::FLOAT);
    REQUIRE(ctx.get_intermediate_data_type() == DataType_t::DOUBLE);
    REQUIRE(ctx.get_io_data_type() == DataType_t::INT64);
    REQUIRE(ctx.get_name() == "chain");
  }
}
