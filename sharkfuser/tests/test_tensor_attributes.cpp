// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace fusili;

TEST_CASE("TensorAttr fill_from_context", "[TensorAttr]") {
  Context ctx;
  ctx.setIntermediateDataType(DataType::Float).setIODataType(DataType::Double);

  SECTION("Virtual tensor gets intermediate data type") {
    TensorAttr t;
    t.setIsVirtual(true);
    t.fillFromContext(ctx);
    REQUIRE(t.getDataType() == DataType::Float);
  }

  SECTION("Non-virtual tensor gets IO data type") {
    TensorAttr t;
    t.setIsVirtual(false);
    t.fillFromContext(ctx);
    REQUIRE(t.getDataType() == DataType::Double);
  }

  SECTION("Already set data type is not changed") {
    TensorAttr t;
    t.setDataType(DataType::Int32);
    t.fillFromContext(ctx);
    REQUIRE(t.getDataType() == DataType::Int32);
  }
}

TEST_CASE("TensorAttr method chaining", "[TensorAttr]") {
  TensorAttr t;
  auto &result = t.setName("test")
                     .setDataType(DataType::Float)
                     .setDim({2, 3})
                     .setStride({3, 1})
                     .setIsVirtual(true)
                     .setUid(42);

  REQUIRE(&result == &t); // Verify chaining returns same object
  REQUIRE(t.getName() == "test");
  REQUIRE(t.getDataType() == DataType::Float);
  REQUIRE(t.getDim() == std::vector<int64_t>{2, 3});
  REQUIRE(t.getStride() == std::vector<int64_t>{3, 1});
  REQUIRE(t.isVirtual());
  REQUIRE(t.getUid() == 42);
}

TEST_CASE("TensorAttr validation edge cases", "[TensorAttr]") {
  SECTION("Empty dim fails validation") {
    TensorAttr t;
    t.setName("nodim").setStride({1});
    REQUIRE(t.validate().isFailure());
  }

  SECTION("Empty stride fails validation") {
    TensorAttr t;
    t.setName("nostride").setDim({1});
    REQUIRE(t.validate().isFailure());
  }

  SECTION("Empty name still validates if dims and strides are set") {
    TensorAttr t;
    t.setDim({2}).setStride({1});
    REQUIRE(t.validate().isOk());
  }

  SECTION("Dim and stride of different ranks is invalid") {
    TensorAttr t;
    t.setName("diffrank").setDim({2}).setStride({1, 1});
    REQUIRE(t.validate().isFailure());
  }

  SECTION("Single dimension tensor") {
    TensorAttr t;
    t.setName("single").setDim({5}).setStride({1});
    REQUIRE(t.validate().isOk());
    REQUIRE(t.getVolume() == 5);
  }

  SECTION("Zero dimension in tensor") {
    TensorAttr t;
    t.setName("zero").setDim({2, 0, 3}).setStride({6, 3, 1});
    REQUIRE(t.validate().isOk());
    REQUIRE(t.getVolume() == 0);
  }

  SECTION("Non-contiguous (strided) tensors fail validation") {
    TensorAttr t1, t2;

    t1.setName("contig").setDim({4, 3}).setStride({3, 1});
    REQUIRE(t1.validate().isOk());

    t2.setName("non_contig").setDim({4, 3}).setStride({1, 4});
    REQUIRE(t2.validate().isFailure());
  }

  SECTION("Virtual and scalar tensors can't coexist") {
    TensorAttr t;
    t.setDim({1}).setStride({1});
    t.setIsVirtual(true).setIsScalar(true);
    REQUIRE(t.validate().isFailure());
  }

  SECTION("Scalar value set but not marked scalar") {
    TensorAttr t(3.14);
    REQUIRE(t.isScalar());
    t.setIsScalar(false);
    REQUIRE(!t.isScalar());
    REQUIRE(t.validate().isFailure());
  }

  SECTION("Scalar value not set but marked scalar") {
    TensorAttr t;
    t.setDim({1}).setStride({1});
    REQUIRE(!t.isScalar());
    t.setIsScalar(true);
    REQUIRE(t.isScalar());
    REQUIRE(t.validate().isFailure());
  }
}

TEST_CASE("TensorAttr scalar value variants", "[TensorAttr]") {
  SECTION("Float scalar") {
    TensorAttr t(3.14f);
    auto val = t.getScalarValue();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<float>(val.value()));
    REQUIRE(std::get<float>(val.value()) == 3.14f);
    REQUIRE(t.getDim() == std::vector<int64_t>{1});
    REQUIRE(t.getStride() == std::vector<int64_t>{1});
    REQUIRE(t.isScalar());
  }

  SECTION("Double scalar") {
    TensorAttr t(2.718);
    auto val = t.getScalarValue();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<double>(val.value()));
    REQUIRE(std::get<double>(val.value()) == 2.718);
    REQUIRE(t.getDim() == std::vector<int64_t>{1});
    REQUIRE(t.getStride() == std::vector<int64_t>{1});
    REQUIRE(t.isScalar());
  }

  SECTION("Int32 scalar") {
    TensorAttr t(int32_t(-42));
    auto val = t.getScalarValue();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<int32_t>(val.value()));
    REQUIRE(std::get<int32_t>(val.value()) == -42);
    REQUIRE(t.getDim() == std::vector<int64_t>{1});
    REQUIRE(t.getStride() == std::vector<int64_t>{1});
    REQUIRE(t.isScalar());
  }

  SECTION("Int64 scalar") {
    TensorAttr t(int64_t(-123456789));
    auto val = t.getScalarValue();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<int64_t>(val.value()));
    REQUIRE(std::get<int64_t>(val.value()) == -123456789);
    REQUIRE(t.getDim() == std::vector<int64_t>{1});
    REQUIRE(t.getStride() == std::vector<int64_t>{1});
    REQUIRE(t.isScalar());
  }
}

TEST_CASE("TensorAttr UID management", "[TensorAttr]") {
  TensorAttr t;
  REQUIRE(!t.hasUid());
  REQUIRE(t.getUid() == 0);

  t.setUid(0); // Setting to 0 should still mark as assigned
  REQUIRE(t.hasUid());
  REQUIRE(t.getUid() == 0);

  t.setUid(100);
  REQUIRE(t.hasUid());
  REQUIRE(t.getUid() == 100);

  t.clearUid();
  REQUIRE(!t.hasUid());
  REQUIRE(t.getUid() == 0);
}

TEST_CASE("TensorAttr output vs virtual", "[TensorAttr]") {
  TensorAttr t;

  t.setOutput(true);
  REQUIRE(!t.isVirtual());

  t.setOutput(false);
  REQUIRE(t.isVirtual());

  t.setIsVirtual(false);
  REQUIRE(!t.isVirtual());

  t.setIsVirtual(true);
  REQUIRE(t.isVirtual());
}
