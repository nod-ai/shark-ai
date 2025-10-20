// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

namespace {
// Based on parameters, generates a unique name for the graph
std::string generate_name(PointwiseAttr::Mode mode, DataType type,
                          const std::vector<std::vector<int64_t>> &dims) {
  std::string name =
      std::format("pointwise_{}_", PointwiseAttr::modeToStr.at(mode));
  for (size_t i = 0; i < dims.size(); ++i) {
    name += std::format("in{}_", i);
    for (size_t j = 0; j < dims[i].size(); ++j) {
      name += std::format("{}_", dims[i][j]);
    }
  }
  name += std::format("dt{}", DataTypeToMlirTypeAsm.at(type));
  return name;
};

// Generator of initial values for input buffers
template <typename T> std::vector<T> generateInput(PointwiseAttr::Mode mode) {
  switch (mode) {
  case PointwiseAttr::Mode::ADD:
  case PointwiseAttr::Mode::DIV:
  case PointwiseAttr::Mode::MUL:
  case PointwiseAttr::Mode::SUB:
    if constexpr (std::is_same_v<T, float>) {
      return std::vector<float>{-100.5, -20.0};
    }
    if constexpr (std::is_same_v<T, int>) {
      return std::vector<int>{-50, -10};
    }
    if constexpr (std::is_same_v<T, half>) {
      return std::vector<half>{-32.5f16, 2.f16};
    }
    if constexpr (std::is_same_v<T, int16_t>) {
      return std::vector<int16_t>{-5, -2};
    }
    if constexpr (std::is_same_v<T, int8_t>) {
      return std::vector<int8_t>{-7, 2};
    }
    FAIL("Unsupported data type");
  case PointwiseAttr::Mode::RELU_FWD:
    return std::vector<T>{-10};
  default:
    FAIL("Unsupported pointwise mode: " << PointwiseAttr::modeToStr.at(mode));
  }
  return std::vector<T>();
}

} // namespace

TEST_CASE("Pointwise ops", "[pointwise][graph]") {
  const auto mode =
      GENERATE(PointwiseAttr::Mode::ADD, PointwiseAttr::Mode::DIV,
               PointwiseAttr::Mode::MUL, PointwiseAttr::Mode::RELU_FWD,
               PointwiseAttr::Mode::SUB);

  const auto inCount = PointwiseAttr::modeToRequiredInputCount.at(mode);
  // Currently the test supports only one or two inputs
  REQUIRE(inCount <= 2);

  std::vector<std::vector<int64_t>> dims(inCount, {2, 16, 64, 64});
  if (inCount > 1) {
    dims[1] = GENERATE(std::vector<int64_t>{2, 16, 64, 64},
                       std::vector<int64_t>{1, 16, 1, 1},
                       std::vector<int64_t>{1, 1, 64, 64});
  }

  auto execute = [&]<typename T>(const std::shared_ptr<Handle> &handlePtr,
                                 DataType dt, const std::vector<T> &xi) {
    auto build_new_graph = [&](const Handle &handle) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generate_name(mode, dt, dims));
      graph->setIODataType(dt).setComputeDataType(dt);

      // Validate passed input count
      REQUIRE(xi.size() == inCount);

      // Initialize input tensors
      std::vector<std::shared_ptr<TensorAttr>> Xi(dims.size());
      for (size_t i = 0; i < dims.size(); ++i) {
        Xi[i] = graph->tensor(
            TensorAttr()
                .setName("in" + std::to_string(i))
                .setDim(dims[i])
                .setStride(generateStrideFromDim(
                    dims[i], getContiguousStrideOrder(dims[i].size()))));
      }

      // Create Pointwise op
      auto pointwiseAttr = PointwiseAttr().setMode(mode);
      std::shared_ptr<TensorAttr> pointwiseResult;
      if (Xi.size() == 2) {
        pointwiseResult = graph->pointwise(Xi[0], Xi[1], pointwiseAttr);
      } else if (Xi.size() == 1) {
        pointwiseResult = graph->pointwise(Xi[0], pointwiseAttr);
      } else {
        FAIL("Unsupported input count: " << xi.size());
      }

      pointwiseResult->setName("result").setOutput(true);

      // Validate, infer missing properties
      REQUIRE(isOk(graph->validate()));

      // Compile
      REQUIRE(isOk(graph->compile(handle, /*remove=*/true)));

      return std::make_tuple(graph, Xi, pointwiseResult);
    };

    Handle &handle = *handlePtr;
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, Xi, Y] = build_new_graph(handle);

    // Allocate input buffers.
    std::vector<std::shared_ptr<Buffer>> xiBufs(Xi.size());
    for (size_t i = 0; i < Xi.size(); i++) {
      xiBufs[i] = FUSILLI_REQUIRE_UNWRAP(
          allocateBufferOfType(handle, Xi[i], dt, xi[i]));
    }

    // Allocate output buffer.
    auto yBuf =
        FUSILLI_REQUIRE_UNWRAP(allocateBufferOfType(handle, Y, dt, 0.0f));

    // Create variant pack.
    std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
        variantPack;
    for (size_t i = 0; i < xi.size(); i++) {
      variantPack.insert({Xi[i], xiBufs[i]});
    }
    variantPack.insert({Y, yBuf});

    // Execute graph once.
    REQUIRE(isOk(graph->execute(variantPack)));

    // Calculate reference value
    T y = 0;
    switch (mode) {
    case PointwiseAttr::Mode::ADD: {
      y = xi[0] + xi[1];
      break;
    }
    case PointwiseAttr::Mode::DIV: {
      y = xi[0] / xi[1];
      break;
    }
    case PointwiseAttr::Mode::MUL: {
      y = xi[0] * xi[1];
      break;
    }
    case PointwiseAttr::Mode::RELU_FWD: {
      y = std::max(xi[0], T(0));
      break;
    }
    case PointwiseAttr::Mode::SUB: {
      y = xi[0] - xi[1];
      break;
    }
    default:
      FAIL("Unsupported pointwise mode: " << PointwiseAttr::modeToStr.at(mode));
    }

    // Read output buffers.
    std::vector<T> result;
    REQUIRE(isOk(yBuf->read(handle, result)));
    for (auto val : result)
      REQUIRE(val == y);

    // Execute graph a few times.
    constexpr size_t numIters = 1;
    for (size_t i = 0; i < numIters; i++)
      REQUIRE(isOk(graph->execute(variantPack)));

    // Repeat output buffer checks.
    result.clear();
    REQUIRE(isOk(yBuf->read(handle, result)));
    for (auto val : result)
      REQUIRE(val == y);
  };

  // Parameterize sample by backend and create device-specific handles.
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
  }
#endif

  // fp32
  execute(handlePtr, DataType::Float, generateInput<float>(mode));
  // int32
  execute(handlePtr, DataType::Int32, generateInput<int>(mode));
  // fp16
  execute(handlePtr, DataType::Half, generateInput<half>(mode));
  // int16
  execute(handlePtr, DataType::Int16, generateInput<int16_t>(mode));
  // int8
  execute(handlePtr, DataType::Int8, generateInput<int8_t>(mode));
}
