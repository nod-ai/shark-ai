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
  attr.setName("foo");
  REQUIRE(attr.getName() == "foo");

  attr.setComputeDataType(DataType_t::FLOAT);
  REQUIRE(attr.computeDataType == DataType_t::FLOAT);
}

TEST_CASE("AttributesCRTP set/get input/output tensors", "[attributes_crtp]") {
  DummyAttr attr;
  auto tensor_in = std::make_shared<TensorAttr>(1.0f);
  auto tensor_out = std::make_shared<TensorAttr>(2.0f);

  attr.setInput("in", tensor_in);
  attr.setOutput("out", tensor_out);

  REQUIRE(attr.getInput("in") == tensor_in);
  REQUIRE(attr.getOutput("out") == tensor_out);
  REQUIRE(attr.getInput("missing") == nullptr);
  REQUIRE(attr.getOutput("missing") == nullptr);
}

TEST_CASE(
    "AttributesCRTP fill_from_context sets compute_data_type and fills tensors",
    "[attributes_crtp]") {
  DummyAttr attr;
  auto in = std::make_shared<TensorAttr>(2.0f);
  auto out = std::make_shared<TensorAttr>();
  attr.setInput("in", in);
  attr.setOutput("out", out);

  REQUIRE(attr.computeDataType == DataType_t::NOT_SET);
  REQUIRE(attr.getInput("in")->getDataType() == DataType_t::FLOAT);
  REQUIRE(attr.getOutput("out")->getDataType() == DataType_t::NOT_SET);

  Context ctx;
  ctx.setComputeDataType(DataType_t::DOUBLE)
      .setIntermediateDataType(DataType_t::FLOAT)
      .setIODataType(DataType_t::INT32);

  attr.fillFromContext(ctx);
  REQUIRE(attr.computeDataType == DataType_t::DOUBLE);
  REQUIRE(attr.getInput("in")->getDataType() == DataType_t::FLOAT);
  REQUIRE(attr.getOutput("out")->getDataType() == DataType_t::INT32);
}
