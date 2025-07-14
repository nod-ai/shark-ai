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
  ctx.set_intermediate_data_type(DataType_t::FLOAT)
      .set_io_data_type(DataType_t::DOUBLE);

  SECTION("Virtual tensor gets intermediate data type") {
    TensorAttr t;
    t.set_is_virtual(true);
    t.fill_from_context(ctx);
    REQUIRE(t.get_data_type() == DataType_t::FLOAT);
  }

  SECTION("Non-virtual tensor gets IO data type") {
    TensorAttr t;
    t.set_is_virtual(false);
    t.fill_from_context(ctx);
    REQUIRE(t.get_data_type() == DataType_t::DOUBLE);
  }

  SECTION("Already set data type is not changed") {
    TensorAttr t;
    t.set_data_type(DataType_t::INT32);
    t.fill_from_context(ctx);
    REQUIRE(t.get_data_type() == DataType_t::INT32);
  }
}

TEST_CASE("TensorAttr method chaining", "[TensorAttr]") {
  TensorAttr t;
  auto &result = t.set_name("test")
                     .set_data_type(DataType_t::FLOAT)
                     .set_dim({2, 3})
                     .set_stride({3, 1})
                     .set_is_virtual(true)
                     .set_uid(42);

  REQUIRE(&result == &t); // Verify chaining returns same object
  REQUIRE(t.get_name() == "test");
  REQUIRE(t.get_data_type() == DataType_t::FLOAT);
  REQUIRE(t.get_dim() == std::vector<int64_t>{2, 3});
  REQUIRE(t.get_stride() == std::vector<int64_t>{3, 1});
  REQUIRE(t.get_is_virtual());
  REQUIRE(t.get_uid() == 42);
}

TEST_CASE("TensorAttr validation edge cases", "[TensorAttr]") {
  SECTION("Empty dim fails validation") {
    TensorAttr t;
    t.set_stride({1});
    REQUIRE(t.validate().is_failure());
  }

  SECTION("Empty stride fails validation") {
    TensorAttr t;
    t.set_dim({1});
    REQUIRE(t.validate().is_failure());
  }

  SECTION("Empty name still validates if dims and strides are set") {
    TensorAttr t;
    t.set_dim({2}).set_stride({1});
    REQUIRE(t.validate().is_ok());
  }

  SECTION("Dim and stride of different ranks is invalid") {
    TensorAttr t;
    t.set_dim({2}).set_stride({1, 1});
    REQUIRE(t.validate().is_failure());
  }

  SECTION("Single dimension tensor") {
    TensorAttr t;
    t.set_name("single").set_dim({5}).set_stride({1});
    REQUIRE(t.validate().is_ok());
    REQUIRE(t.get_volume() == 5);
  }

  SECTION("Zero dimension in tensor") {
    TensorAttr t;
    t.set_name("zero").set_dim({2, 0, 3}).set_stride({0, 0, 1});
    REQUIRE(t.validate().is_ok());
    REQUIRE(t.get_volume() == 0);
  }

  SECTION("Virtual and scalar tensors can't coexist") {
    TensorAttr t;
    t.set_dim({1}).set_stride({1});
    t.set_is_virtual(true).set_is_scalar(true);
    REQUIRE(t.validate().is_failure());
  }

  SECTION("Scalar value set but not marked scalar") {
    TensorAttr t(3.14);
    REQUIRE(t.get_is_scalar());
    t.set_is_scalar(false);
    REQUIRE(!t.get_is_scalar());
    REQUIRE(t.validate().is_failure());
  }

  SECTION("Scalar value not set but marked scalar") {
    TensorAttr t;
    t.set_dim({1}).set_stride({1});
    REQUIRE(!t.get_is_scalar());
    t.set_is_scalar(true);
    REQUIRE(t.get_is_scalar());
    REQUIRE(t.validate().is_failure());
  }
}

TEST_CASE("TensorAttr scalar value variants", "[TensorAttr]") {
  SECTION("Float scalar") {
    TensorAttr t(3.14f);
    auto val = t.get_scalar_value();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<float>(val.value()));
    REQUIRE(std::get<float>(val.value()) == 3.14f);
    REQUIRE(t.get_dim() == std::vector<int64_t>{1});
    REQUIRE(t.get_stride() == std::vector<int64_t>{1});
    REQUIRE(t.get_is_scalar());
  }

  SECTION("Double scalar") {
    TensorAttr t(2.718);
    auto val = t.get_scalar_value();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<double>(val.value()));
    REQUIRE(std::get<double>(val.value()) == 2.718);
    REQUIRE(t.get_dim() == std::vector<int64_t>{1});
    REQUIRE(t.get_stride() == std::vector<int64_t>{1});
    REQUIRE(t.get_is_scalar());
  }

  SECTION("Int32 scalar") {
    TensorAttr t(int32_t(-42));
    auto val = t.get_scalar_value();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<int32_t>(val.value()));
    REQUIRE(std::get<int32_t>(val.value()) == -42);
    REQUIRE(t.get_dim() == std::vector<int64_t>{1});
    REQUIRE(t.get_stride() == std::vector<int64_t>{1});
    REQUIRE(t.get_is_scalar());
  }

  SECTION("Int64 scalar") {
    TensorAttr t(int64_t(-123456789));
    auto val = t.get_scalar_value();
    REQUIRE(val.has_value());
    REQUIRE(std::holds_alternative<int64_t>(val.value()));
    REQUIRE(std::get<int64_t>(val.value()) == -123456789);
    REQUIRE(t.get_dim() == std::vector<int64_t>{1});
    REQUIRE(t.get_stride() == std::vector<int64_t>{1});
    REQUIRE(t.get_is_scalar());
  }
}

TEST_CASE("TensorAttr UID management", "[TensorAttr]") {
  TensorAttr t;
  REQUIRE(!t.has_uid());
  REQUIRE(t.get_uid() == 0);

  t.set_uid(0); // Setting to 0 should still mark as assigned
  REQUIRE(t.has_uid());
  REQUIRE(t.get_uid() == 0);

  t.set_uid(100);
  REQUIRE(t.has_uid());
  REQUIRE(t.get_uid() == 100);

  t.clear_uid();
  REQUIRE(!t.has_uid());
  REQUIRE(t.get_uid() == 0);
}

TEST_CASE("TensorAttr output vs virtual", "[TensorAttr]") {
  TensorAttr t;

  t.set_output(true);
  REQUIRE(!t.get_is_virtual());

  t.set_output(false);
  REQUIRE(t.get_is_virtual());

  t.set_is_virtual(false);
  REQUIRE(!t.get_is_virtual());

  t.set_is_virtual(true);
  REQUIRE(t.get_is_virtual());
}
