// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace fusilli;

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
                     .setIsVirtual(true);

  REQUIRE(&result == &t); // Verify chaining returns same object
  REQUIRE(t.getName() == "test");
  REQUIRE(t.getDataType() == DataType::Float);
  REQUIRE(t.getDim() == std::vector<int64_t>{2, 3});
  REQUIRE(t.getStride() == std::vector<int64_t>{3, 1});
  REQUIRE(t.isVirtual());
}

TEST_CASE("TensorAttr validation edge cases", "[TensorAttr]") {
  SECTION("Unspecified dim fails validation") {
    TensorAttr t;
    t.setName("nodim").setStride({1}).setDataType(DataType::Float);
    auto status = t.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Tensor 'nodim' dims not set");
  }

  SECTION("Unspecified stride fails validation") {
    TensorAttr t;
    t.setName("nostride").setDim({1}).setDataType(DataType::Float);
    auto status = t.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Tensor 'nostride' strides not set");
  }

  SECTION("Unspecified dtype fails validation") {
    TensorAttr t;
    t.setName("nostride").setDim({1}).setStride({1});
    auto status = t.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "Tensor 'nostride' data type not set");
  }

  SECTION(
      "Unspecified name still validates if dims, strides and dtype are set") {
    TensorAttr t;
    t.setDim({2}).setStride({1}).setDataType(DataType::Float);
    REQUIRE(isOk(t.validate()));
  }

  SECTION("Dim and stride of different ranks is invalid") {
    TensorAttr t;
    t.setName("diffrank")
        .setDim({2})
        .setStride({1, 1})
        .setDataType(DataType::Float);
    auto status = t.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "Tensor 'diffrank' uses dim and stride of different dimensionality");
  }

  SECTION("Single dimension tensor") {
    TensorAttr t;
    t.setName("single").setDim({5}).setStride({1}).setDataType(DataType::Float);
    REQUIRE(isOk(t.validate()));
    REQUIRE(t.getVolume() == 5);
  }

  SECTION("Zero dimension tensor") {
    TensorAttr t;
    t.setName("zero").setDim({2, 0, 3}).setStride({6, 3, 1}).setDataType(
        DataType::Float);
    REQUIRE(isOk(t.validate()));
    REQUIRE(t.getVolume() == 0);
  }

  SECTION("Non-contiguous (strided) tensors fail validation") {
    TensorAttr t1, t2;

    t1.setName("contig").setDim({4, 3}).setStride({3, 1}).setDataType(
        DataType::Float);
    REQUIRE(isOk(t1.validate()));

    t2.setName("non_contig")
        .setDim({4, 3})
        .setStride({1, 4})
        .setDataType(DataType::Float);
    auto status = t2.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::NotImplemented);
    REQUIRE(
        status.getMessage() ==
        "Tensor 'non_contig' is not contiguous as defined by its stride; "
        "please specify a stride {A, B, ... Z} where A > B > ... Z and Z == 1. "
        "This will be supported in a future release");
  }

  SECTION("Virtual and scalar tensors can't coexist") {
    TensorAttr t;
    t.setName("invalid").setDim({1}).setStride({1}).setDataType(
        DataType::Float);
    t.setIsVirtual(true).setIsScalar(true);
    auto status = t.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "Tensor 'invalid' cannot be both virtual "
                                   "(intermediate) and a scalar constant");
  }

  SECTION("Scalar value set but not marked scalar") {
    TensorAttr t(3.14);
    REQUIRE(t.isScalar());
    t.setName("nonscalar").setIsScalar(false);
    REQUIRE(!t.isScalar());
    auto status = t.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "Tensor 'nonscalar' has a scalar value set "
                                   "but is not marked as a scalar");
  }

  SECTION("Scalar value not set but marked scalar") {
    TensorAttr t;
    t.setName("nonscalar")
        .setDim({1})
        .setStride({1})
        .setDataType(DataType::Float);
    REQUIRE(!t.isScalar());
    t.setIsScalar(true);
    REQUIRE(t.isScalar());
    auto status = t.validate();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() == "Tensor 'nonscalar' is marked as a scalar "
                                   "but does not have a scalar value set");
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
