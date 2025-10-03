// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <CLI/CLI.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// For CLI11 Range Validators
const auto NonNegativeInteger =
    CLI::Range(int64_t{0}, std::numeric_limits<int64_t>::max());
const auto PositiveInteger =
    CLI::Range(int64_t{1}, std::numeric_limits<int64_t>::max());

ErrorObject benchmark_conv_fprop(int64_t n, int64_t c, int64_t h, int64_t w,
                                 int64_t k, int64_t r, int64_t s, int64_t u,
                                 int64_t v, int64_t p, int64_t q, int64_t l,
                                 int64_t j) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::GFX942));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Build graph for the given handle (device), validate and compile it.
  auto graph = std::make_shared<Graph>();
  graph->setName("benchmark_conv_fprop");
  graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

  auto X = graph->tensor(TensorAttr()
                             .setName("image")
                             .setDim({n, c, h, w})
                             .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto W = graph->tensor(TensorAttr()
                             .setName("filter")
                             .setDim({k, c, r, s})
                             .setStride({c * r * s, r * s, s, 1})); // KCRS

  auto conv_attr = ConvFPropAttr()
                       .setStride({u, v})
                       .setPadding({p, q})
                       .setDilation({l, j})
                       .setName("conv_fprop");

  auto Y = graph->convFProp(X, W, conv_attr);
  Y->setOutput(true);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph->validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));

  // Allocate input buffer.
  auto xBuf = std::make_shared<Buffer>(FUSILLI_TRY(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({n, c, h, w}),
                       /*data=*/std::vector<half>(n * c * h * w, half(1.0f)))));

  // Allocate weight buffer.
  auto wBuf = std::make_shared<Buffer>(FUSILLI_TRY(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({k, c, r, s}),
                       /*data=*/std::vector<half>(k * c * r * s, half(1.0f)))));

  // Allocate output buffer.
  auto yBuf = std::make_shared<Buffer>(FUSILLI_TRY(
      Buffer::allocate(handle,
                       /*shape=*/castToSizeT({n, k, h, w}),
                       /*data=*/std::vector<half>(n * k * h * w, half(0.0f)))));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  // Execute graph a few times.
  for (size_t i = 0; i < 100; i++)
    FUSILLI_CHECK_ERROR(graph->execute(variantPack));

  return ok();
}

int main(int argc, char **argv) {
  CLI::App mainApp{"Fusilli Benchmark Driver"};
  mainApp.require_subcommand(1);

  // Conv flags are kept in sync with MIOpen's ConvDriver:
  // https://github.com/ROCm/rocm-libraries/blob/db0544fb61f2c7bd5a86dce98d4963420c1c741a/projects/miopen/driver/conv_driver.hpp#L878
  CLI::App *convApp =
      mainApp.add_subcommand("conv", "Fusilli Benchmark Forward Convolution");
  int64_t n, c, h, w, k, r, s, u, v, p, q, l, j;
  convApp->add_option("--batchsize,-n", n, "Input batch size")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_channels,-c", c, "Input channels")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_h,-H", h, "Input height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_w,-W", w, "Input width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--out_channels,-k", k, "Output channels")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--fil_h,-y", r, "Filter height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--fil_w,-x", s, "Filter width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_h,-u", u, "Conv stride height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_w,-v", v, "Conv stride width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--pad_h,-p", p, "Conv padding height")
      ->required()
      ->check(NonNegativeInteger);
  convApp->add_option("--pad_w,-q", q, "Conv padding width")
      ->required()
      ->check(NonNegativeInteger);
  convApp->add_option("--dilation_h,-l", l, "Conv dilation height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--dilation_w,-j", j, "Conv dilation width")
      ->required()
      ->check(PositiveInteger);

  CLI11_PARSE(mainApp, argc, argv);

  std::cout << "Fusilli Benchmark started..." << std::endl;

  if (convApp->parsed()) {
    auto status = benchmark_conv_fprop(n, c, h, w, k, r, s, u, v, p, q, l, j);
    if (isError(status)) {
      std::cerr << "Fusilli Benchmark failed: " << status << std::endl;
      return 1;
    }
  }

  std::cout << "Fusilli Benchmark complete!" << std::endl;
  return 0;
}
