// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <span>
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

TEST_CASE("TensorAttr setter templated overrides", "[TensorAttr]") {
  TensorAttr t;
  std::vector<int64_t> dimVec = {2, 3, 4};
  std::vector<int64_t> strideVec = {12, 4, 1};

  std::span<int64_t> dimSpan(dimVec);
  std::span<int64_t> strideSpan(strideVec);

  // Setters either take a const std::vector& or a type constrained template,
  // std::span should call the templated override.
  auto &result = t.setDim(dimSpan).setStride(strideSpan);

  REQUIRE(t.getDim() == dimVec);
  REQUIRE(t.getStride() == strideVec);
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

TEST_CASE("TensorAttr isContiguous and isChannelsLast checks", "[TensorAttr]") {
  TensorAttr t1;
  t1.setName("contiguous_tensor")
      .setDataType(DataType::Float)
      .setDim({2, 3, 4})
      .setStride({12, 4, 1});
  REQUIRE(t1.isContiguous());
  REQUIRE(!t1.isChannelsLast());
  REQUIRE(t1.getDim() == std::vector<int64_t>{2, 3, 4});
  REQUIRE(t1.getPhysicalDim() == std::vector<int64_t>{2, 3, 4});

  TensorAttr t2;
  t2.setName("channels_last_tensor")
      .setDataType(DataType::Float)
      .setDim({2, 3, 4})
      .setStride({12, 1, 3});
  REQUIRE(!t2.isContiguous());
  REQUIRE(t2.isChannelsLast());
  REQUIRE(t2.getDim() == std::vector<int64_t>{2, 3, 4});
  REQUIRE(t2.getPhysicalDim() == std::vector<int64_t>{2, 4, 3});
}

TEST_CASE("Stride order utils", "[TensorAttr utils]") {
  // Contiguous (channels-first) stride order
  REQUIRE(getContiguousStrideOrder(3) == std::vector<size_t>({2, 1, 0}));
  REQUIRE(getContiguousStrideOrder(4) == std::vector<size_t>({3, 2, 1, 0}));
  REQUIRE(getContiguousStrideOrder(5) == std::vector<size_t>({4, 3, 2, 1, 0}));

  // Channels-last stride order
  REQUIRE(getChannelsLastStrideOrder(3) == std::vector<size_t>({2, 0, 1}));
  REQUIRE(getChannelsLastStrideOrder(4) == std::vector<size_t>({3, 0, 2, 1}));
  REQUIRE(getChannelsLastStrideOrder(5) ==
          std::vector<size_t>({4, 0, 3, 2, 1}));

  // Generate stride from dim and stride order
  REQUIRE(generateStrideFromDim({10, 3, 12, 12}, {3, 0, 2, 1}) ==
          std::vector<int64_t>({432, 1, 36, 3}));
  REQUIRE(generateStrideFromDim({10, 3, 12, 12}, {3, 2, 1, 0}) ==
          std::vector<int64_t>({432, 144, 12, 1}));

  // Ambiguous case (multiple dims of size 1)
  REQUIRE(generateStrideFromDim({256, 128, 1, 1}, {3, 0, 2, 1}) ==
          std::vector<int64_t>({128, 1, 128, 128}));
  REQUIRE(generateStrideFromDim({256, 128, 1, 1}, {3, 2, 1, 0}) ==
          std::vector<int64_t>({128, 1, 1, 1}));
}

TEST_CASE("Permute order utils", "[TensorAttr utils]") {
  // Preserve contiguous permute order
  REQUIRE(getPreserveContiguousPermuteOrder(1) == std::vector<int64_t>({0}));
  REQUIRE(getPreserveContiguousPermuteOrder(2) == std::vector<int64_t>({0, 1}));
  REQUIRE(getPreserveContiguousPermuteOrder(3) ==
          std::vector<int64_t>({0, 1, 2}));
  REQUIRE(getPreserveContiguousPermuteOrder(4) ==
          std::vector<int64_t>({0, 1, 2, 3}));
  REQUIRE(getPreserveContiguousPermuteOrder(5) ==
          std::vector<int64_t>({0, 1, 2, 3, 4}));

  // Channels-last to contiguous permute order
  REQUIRE(getChannelsLastToContiguousPermuteOrder(3) ==
          std::vector<int64_t>({0, 2, 1}));
  REQUIRE(getChannelsLastToContiguousPermuteOrder(4) ==
          std::vector<int64_t>({0, 3, 1, 2}));
  REQUIRE(getChannelsLastToContiguousPermuteOrder(5) ==
          std::vector<int64_t>({0, 4, 1, 2, 3}));

  // Contiguous to channels-last permute order
  REQUIRE(getContiguousToChannelsLastPermuteOrder(3) ==
          std::vector<int64_t>({0, 2, 1}));
  REQUIRE(getContiguousToChannelsLastPermuteOrder(4) ==
          std::vector<int64_t>({0, 2, 3, 1}));
  REQUIRE(getContiguousToChannelsLastPermuteOrder(5) ==
          std::vector<int64_t>({0, 2, 3, 4, 1}));
}

TEST_CASE("computeBroadcastShapes", "[TensorAttr utils]") {
  SECTION("Empty inputs") {
    SECTION("All shapes empty") {
      std::vector<std::vector<int64_t>> shapes = {{}, {}, {}};
      auto result = computeBroadcastShape(shapes);
      ErrorObject err = result;
      REQUIRE(isError(err));
      REQUIRE(err.getCode() == ErrorCode::InvalidAttribute);
      REQUIRE(err.getMessage() == "All input shapes are empty");
    }

    SECTION("No shapes") {
      std::vector<std::vector<int64_t>> shapes = {};
      auto result = computeBroadcastShape(shapes);
      ErrorObject err = result;
      REQUIRE(isError(err));
      REQUIRE(err.getCode() == ErrorCode::InvalidAttribute);
      REQUIRE(err.getMessage() == "All input shapes are empty");
    }
  }

  SECTION("Single shape") {
    SECTION("Single 1D shape") {
      std::vector<std::vector<int64_t>> shapes = {{5}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{5});
    }

    SECTION("Single 4D shape") {
      std::vector<std::vector<int64_t>> shapes = {{16, 32, 64, 128}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{16, 32, 64, 128});
    }
  }

  SECTION("Identical shapes") {
    SECTION("Two identical shapes") {
      std::vector<std::vector<int64_t>> shapes = {{3, 4}, {3, 4}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{3, 4});
    }

    SECTION("Three identical shapes") {
      std::vector<std::vector<int64_t>> shapes = {
          {2, 3, 4}, {2, 3, 4}, {2, 3, 4}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{2, 3, 4});
    }
  }

  SECTION("Different rank shapes") {
    SECTION("1D + 2D") {
      std::vector<std::vector<int64_t>> shapes = {{4}, {3, 4}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{3, 4});
    }

    SECTION("2D + 3D") {
      std::vector<std::vector<int64_t>> shapes = {{4, 5}, {2, 4, 5}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{2, 4, 5});
    }

    SECTION("1D + 2D + 3D") {
      std::vector<std::vector<int64_t>> shapes = {{5}, {3, 5}, {2, 3, 5}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{2, 3, 5});
    }
  }

  SECTION("Unit dimension broadcasting") {
    SECTION("Broadcasting with 1s") {
      std::vector<std::vector<int64_t>> shapes = {{1, 4}, {3, 1}, {3, 1}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{3, 4});
    }

    SECTION("Complex broadcasting - PyTorch style") {
      std::vector<std::vector<int64_t>> shapes = {{16, 32, 64, 128},
                                                  {1, 32, 1, 1}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{16, 32, 64, 128});
    }

    SECTION("Single element tensors") {
      std::vector<std::vector<int64_t>> shapes = {{1}, {1}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{1});
    }
  }

  SECTION("Incompatible dimensions") {
    SECTION("Incompatible dimensions") {
      std::vector<std::vector<int64_t>> shapes = {{3}, {4}};
      auto result = computeBroadcastShape(shapes);
      ErrorObject err = result;
      REQUIRE(isError(err));
      REQUIRE(err.getCode() == ErrorCode::InvalidAttribute);
      REQUIRE(err.getMessage() == "Cannot broadcast two non unit dimensions");
    }

    SECTION("Complex incompatible case") {
      std::vector<std::vector<int64_t>> shapes = {{2, 3, 4}, {2, 5, 4}};
      auto result = computeBroadcastShape(shapes);
      ErrorObject err = result;
      REQUIRE(isError(err));
      REQUIRE(err.getMessage() == "Cannot broadcast two non unit dimensions");
    }

    SECTION("Unit vs non-unit mismatch") {
      std::vector<std::vector<int64_t>> shapes = {{1, 3}, {2, 4}};
      auto result = computeBroadcastShape(shapes);
      ErrorObject err = result;
      REQUIRE(isError(err));
      REQUIRE(err.getMessage() == "Cannot broadcast two non unit dimensions");
    }
  }

  SECTION("Mixed empty and non-empty") {
    SECTION("Empty shapes filtered out") {
      std::vector<std::vector<int64_t>> shapes = {
          {}, {3, 4}, {1}, {3, 1}, {1, 4}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{3, 4});
    }

    SECTION("All empty except one") {
      std::vector<std::vector<int64_t>> shapes = {{}, {}, {5, 6, 7}, {}};
      auto result = computeBroadcastShape(shapes);
      REQUIRE(isOk(result));
      REQUIRE(*result == std::vector<int64_t>{5, 6, 7});
    }
  }
}

TEST_CASE("generateStrideOrderPreservingFormat", "[TensorAttr utils]") {
  REQUIRE(generateStrideOrderPreservingFormat({12, 4, 1}, 3) ==
          std::vector<size_t>({2, 1, 0}));

  REQUIRE(generateStrideOrderPreservingFormat({432, 1, 36, 3}, 4) ==
          std::vector<size_t>({3, 0, 2, 1}));

  REQUIRE(generateStrideOrderPreservingFormat({1}, 1) ==
          std::vector<size_t>({0}));

  // Input has 3 dimensions, output needs 4
  REQUIRE(generateStrideOrderPreservingFormat({12, 4, 1}, 4) ==
          std::vector<size_t>({2, 1, 0, 3}));

  // When strides are equal, the function should preserve the original order
  REQUIRE(generateStrideOrderPreservingFormat({1, 1, 1}, 3) ==
          std::vector<size_t>({0, 1, 2}));

  // Mixed equal and different strides
  REQUIRE(generateStrideOrderPreservingFormat({6, 1, 1}, 3) ==
          std::vector<size_t>({2, 0, 1}));
}
