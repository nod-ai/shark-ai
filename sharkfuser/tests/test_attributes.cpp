// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <unordered_map>

using namespace fusili;

struct DummyAttr : public AttributesCRTP<DummyAttr> {
  std::unordered_map<std::string, std::shared_ptr<TensorAttr>> inputs;
  std::unordered_map<std::string, std::shared_ptr<TensorAttr>> outputs;
};

TEST_CASE("AttributesCRTP set/get name and compute_data_type",
          "[attributes_crtp]") {
  DummyAttr attr;
  attr.set_name("foo");
  REQUIRE(attr.get_name() == "foo");

  attr.set_compute_data_type(DataType_t::FLOAT);
  REQUIRE(attr.compute_data_type == DataType_t::FLOAT);
}

TEST_CASE("AttributesCRTP set/get input/output tensors", "[attributes_crtp]") {
  DummyAttr attr;
  auto tensor_in = std::make_shared<TensorAttr>(1.0f);
  auto tensor_out = std::make_shared<TensorAttr>(2.0f);

  attr.set_input("in", tensor_in);
  attr.set_output("out", tensor_out);

  REQUIRE(attr.get_input("in") == tensor_in);
  REQUIRE(attr.get_output("out") == tensor_out);
  REQUIRE(attr.get_input("missing") == nullptr);
  REQUIRE(attr.get_output("missing") == nullptr);
}

TEST_CASE(
    "AttributesCRTP fill_from_context sets compute_data_type and fills tensors",
    "[attributes_crtp]") {
  DummyAttr attr;
  auto in = std::make_shared<TensorAttr>(2.0f);
  auto out = std::make_shared<TensorAttr>();
  attr.set_input("in", in);
  attr.set_output("out", out);

  REQUIRE(attr.compute_data_type == DataType_t::NOT_SET);
  REQUIRE(attr.get_input("in")->get_data_type() == DataType_t::FLOAT);
  REQUIRE(attr.get_output("out")->get_data_type() == DataType_t::NOT_SET);

  Context ctx;
  ctx.set_compute_data_type(DataType_t::DOUBLE)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_io_data_type(DataType_t::INT32);

  attr.fill_from_context(ctx);
  REQUIRE(attr.compute_data_type == DataType_t::DOUBLE);
  REQUIRE(attr.get_input("in")->get_data_type() == DataType_t::FLOAT);
  REQUIRE(attr.get_output("out")->get_data_type() == DataType_t::INT32);
}
