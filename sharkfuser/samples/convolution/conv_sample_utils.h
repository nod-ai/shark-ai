// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for fusilli convolution sample tests.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SAMPLES_CONVOLUTION_CONV_SAMPLES_UTILS_H
#define FUSILLI_SAMPLES_CONVOLUTION_CONV_SAMPLES_UTILS_H

#include <fusilli.h>

#include <catch2/catch_all.hpp>

#include <cstdint>
#include <format>
#include <memory>
#include <unordered_map>
#include <vector>

#include "utils.h"

using namespace fusilli;

namespace fusilli_conv_samples {

enum class ConvSampleLayout {
  NCHW_KCRS,
  NHWC_KRSC,
};

struct ConvSampleConfig {
  // Input and filter dims.
  int64_t n; // Batch size
  int64_t c; // Input channels
  int64_t h; // Input height
  int64_t w; // input width
  int64_t k; // Number of filters / output channels
  int64_t r; // Filter height (rows)
  int64_t s; // Filter width (cols)

  // Padding.
  int64_t pad_h = 0;
  int64_t pad_w = 0;

  // Layout.
  ConvSampleLayout layout = ConvSampleLayout::NCHW_KCRS;

  // Pointwise.
  bool bias = false;
  bool relu = false;

  // Expected output.
  half expected = -1.0;
};

inline std::string layoutToString(ConvSampleLayout layout) {
  switch (layout) {
  case ConvSampleLayout::NCHW_KCRS:
    return "nchw_kcrs";
  case ConvSampleLayout::NHWC_KRSC:
    return "nhwc_krsc";
  default:
    return "unknown";
  }
}

struct Sample {
  // Graph to execute.
  Graph graph;
  // Graph inputs.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack;
  // Graph output.
  std::shared_ptr<Buffer> yBuf;
};

inline Sample buildSample(const Handle &handle,
                          const ConvSampleConfig &config) {
  Graph graph;

  // Build graph name.
  std::string name =
      std::format("conv_fprop_sample_{}_n{}_c{}_h{}_w{}_k{}_r{}_s{}",
                  layoutToString(config.layout), config.n, config.c, config.h,
                  config.w, config.k, config.r, config.s);

  if (config.pad_h != 0 || config.pad_w != 0) {
    name += std::format("_pad{}x{}", config.pad_h, config.pad_w);
  }
  if (config.bias) {
    name += "_bias";
  }
  if (config.relu) {
    name += "_relu";
  }
  name += std::format("_{}", handle);

  // Set graph level properties>
  graph.setName(name);
  graph.setIODataType(DataType::Half).setComputeDataType(DataType::Float);

  // Build input and filter.
  std::shared_ptr<TensorAttr> X, W;
  if (config.layout == ConvSampleLayout::NCHW_KCRS) {
    X = graph.tensor(
        TensorAttr()
            .setName("image")
            .setDim({config.n, config.c, config.h, config.w})
            .setStride({config.c * config.h * config.w, config.h * config.w,
                        config.w, 1})); // NCHW

    W = graph.tensor(
        TensorAttr()
            .setName("filter")
            .setDim({config.k, config.c, config.r, config.s})
            .setStride({config.c * config.r * config.s, config.r * config.s,
                        config.s, 1})); // KCRS
  } else {
    X = graph.tensor(TensorAttr()
                         .setName("image")
                         .setDim({config.n, config.c, config.h, config.w})
                         .setStride({config.c * config.h * config.w, 1,
                                     config.c * config.w, config.c})); // NHWC

    W = graph.tensor(TensorAttr()
                         .setName("filter")
                         .setDim({config.k, config.c, config.r, config.s})
                         .setStride({config.c * config.r * config.s, 1,
                                     config.c * config.s, config.c})); // KRSC
  }

  // Build the convolution.
  auto convAttr = ConvFPropAttr()
                      .setPadding({config.pad_h, config.pad_w})
                      .setStride({1, 1})
                      .setDilation({1, 1})
                      .setName("conv_fprop");

  auto Y = graph.convFProp(X, W, convAttr);
  Y->setName("conv_result").setDataType(DataType::Half);

  // Bias.
  std::shared_ptr<TensorAttr> B = nullptr;
  if (config.bias) {
    B = graph.tensor(TensorAttr()
                         .setName("bias")
                         .setDim({1, config.k, 1, 1})
                         .setStride({config.k, 1, config.k, config.k}));
    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    auto biasResult = graph.pointwise(Y, B, biasAttr);
    biasResult->setName("bias_result").setDataType(DataType::Half);

    // Mutate local variable such that Y is always the output.
    Y = biasResult;
  }

  // Relu.
  if (config.relu) {
    auto reluAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::RELU_FWD);
    auto reluResult = graph.pointwise(Y, reluAttr);
    reluResult->setName("relu_result").setDataType(DataType::Half);

    // Mutate local variable such that Y is always the output.
    Y = reluResult;
  }

  // Y should be the output.
  Y->setOutput(true);

  // Validate, infer missing properties.
  FUSILLI_REQUIRE_OK(graph.validate());

  // Compile.
  FUSILLI_REQUIRE_OK(graph.compile(handle, /*remove=*/true));

  // Allocate input buffer.
  auto xBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(X->getPhysicalDim()),
      /*data=*/std::vector<half>(X->getVolume(), half(1.0f)))));

  // Allocate weight buffer.
  auto wBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(W->getPhysicalDim()),
      /*data=*/std::vector<half>(W->getVolume(), half(1.0f)))));

  // Allocate output buffer.
  auto yBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(Y->getPhysicalDim()),
      /*data=*/std::vector<half>(Y->getVolume(), half(0.0f)))));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };
  if (config.bias) {
    // Allocate bias buffer.
    auto bBuf =
        std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(
            handle,
            /*shape=*/castToSizeT(B->getPhysicalDim()),
            /*data=*/std::vector<half>(B->getVolume(), half(1.0f)))));
    // Include bias in variant pack
    variantPack[B] = bBuf;
  }

  return {.graph = std::move(graph),
          .variantPack = std::move(variantPack),
          .yBuf = yBuf};
};

} // namespace fusilli_conv_samples

#endif // FUSILLI_SAMPLES_CONVOLUTION_CONV_SAMPLES_UTILS_H
